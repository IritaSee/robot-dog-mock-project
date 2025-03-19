#!/usr/bin/env python
"""
Script to prepare a dataset for fine-tuning an LLM based on PDF documents.
This script extracts text from PDFs, generates instruction-response pairs,
and formats them into a dataset suitable for LLM fine-tuning.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal
import argparse
from uuid import uuid4

from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_document(
    document_path: str,
    chunking_strategy: str = "by_title",
    strategy: str = "hi_res"
) -> List[Document]:
    """
    Load and process a document from a local path.

    Args:
        document_path (str): The path to the document file on the local system.
        chunking_strategy (str, optional): The strategy to use for chunking the document.
            Defaults to "by_title".
        strategy (str, optional): The strategy to use for processing the document.
            Defaults to "hi_res".

    Returns:
        List[Document]: A list of Document objects processed from the input source.
    """
    logger.info(f"Loading document from local path {document_path}...")
    loader = UnstructuredLoader(
        file_path=document_path,
        chunking_strategy=chunking_strategy,
        strategy=strategy,
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} chunks from {document_path}!")
    return documents


def generate_instruction_pairs(
    documents: List[Document],
    llm_model: Literal["groq", "ollama"],
    num_pairs_per_doc: int = 5,
    temperature: float = 0.7,
) -> List[Dict[str, str]]:
    """
    Generate instruction-response pairs from document chunks using an LLM.

    Args:
        documents (List[Document]): List of document chunks
        llm_model (str): The LLM model to use for generating instruction pairs
        num_pairs_per_doc (int): Number of instruction pairs to generate per document
        temperature (float): Temperature parameter for LLM generation

    Returns:
        List[Dict[str, str]]: List of instruction-response pairs
    """
    logger.info(f"Generating instruction pairs using {llm_model}...")
    
    # Initialize the LLM
    if llm_model == "llama-3.3-70b-versatile":
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=llm_model,
            temperature=temperature,
        )
    elif llm_model == "qwen2.5:14b":
        llm = ChatOllama(
            model=llm_model,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Invalid LLM model: {llm_model}")
    
    # System prompt for generating instruction pairs
    system_prompt = """
    You are an expert at creating high-quality instruction-response pairs for fine-tuning language models.
    Given a text passage about earthquakes, create {num_pairs} different instruction-response pairs.
    
    For each pair:
    1. The instruction should be a natural, diverse question or request that someone might ask about the content
    2. The response should be comprehensive, accurate, and based solely on the provided text
    3. Vary the types of instructions (questions, requests for explanations, summaries, etc.)
    4. Make sure the responses are detailed enough to be useful for training
    
    Format your response as a JSON list of objects with 'instruction' and 'response' fields.
    """
    
    all_pairs = []
    
    for i, doc in enumerate(documents):
        if not doc.page_content.strip():
            continue
            
        try:
            # Create the prompt for generating instruction pairs
            messages = [
                SystemMessage(content=system_prompt.format(num_pairs=num_pairs_per_doc)),
                HumanMessage(content=f"Text passage:\n\n{doc.page_content}\n\nGenerate {num_pairs_per_doc} instruction-response pairs based on this passage.")
            ]
            
            # Generate instruction pairs
            response = llm.invoke(messages)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                json_str = response.content
                # If the response contains markdown code blocks, extract the JSON
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                pairs = json.loads(json_str)
                
                # Validate the format of each pair
                valid_pairs = []
                for pair in pairs:
                    if isinstance(pair, dict) and "instruction" in pair and "response" in pair:
                        # Add metadata
                        pair["id"] = str(uuid4())
                        pair["source_document"] = os.path.basename(doc.metadata.get("source", "unknown"))
                        valid_pairs.append(pair)
                
                all_pairs.extend(valid_pairs)
                logger.info(f"Generated {len(valid_pairs)} valid pairs from document chunk {i+1}/{len(documents)}")
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response for document chunk {i+1}")
                logger.debug(f"Response content: {response.content}")
                
        except Exception as e:
            logger.error(f"Error generating pairs for document chunk {i+1}: {str(e)}")
    
    logger.success(f"Generated a total of {len(all_pairs)} instruction-response pairs")
    return all_pairs


def save_dataset(pairs: List[Dict[str, str]], output_path: str, format: str = "jsonl") -> None:
    """
    Save the instruction-response pairs to a file in the specified format.

    Args:
        pairs (List[Dict[str, str]]): List of instruction-response pairs
        output_path (str): Path to save the dataset
        format (str): Output format (jsonl, csv, or parquet)
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
    elif format in ["csv", "parquet"]:
        df = pd.DataFrame(pairs)
        if format == "csv":
            df.to_csv(output_path, index=False)
        else:  # parquet
            df.to_parquet(output_path, index=False)
    
    logger.success(f"Saved {len(pairs)} pairs to {output_path} in {format} format")


def split_dataset(pairs: List[Dict[str, str]], train_ratio: float = 0.8) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split the dataset into training and validation sets.

    Args:
        pairs (List[Dict[str, str]]): List of instruction-response pairs
        train_ratio (float): Ratio of data to use for training

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Training and validation sets
    """
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Calculate the split point
    split_idx = int(len(pairs) * train_ratio)
    
    # Split the dataset
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    logger.info(f"Split dataset into {len(train_pairs)} training and {len(val_pairs)} validation examples")
    
    return train_pairs, val_pairs


def main():
    """Main function to prepare the fine-tuning dataset."""
    parser = argparse.ArgumentParser(description="Prepare a dataset for fine-tuning an LLM from PDFs")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/finetune",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["jsonl", "csv", "parquet"], 
        default="jsonl",
        help="Output format for the dataset"
    )
    parser.add_argument(
        "--num_pairs_per_chunk", 
        type=int, 
        default=3,
        help="Number of instruction pairs to generate per document chunk"
    )
    parser.add_argument(
        "--chunking_strategy", 
        type=str, 
        default="by_title",
        help="Strategy for chunking documents"
    )
    parser.add_argument(
        "--llm_model", 
        type=str, 
        choices=["llama-3.3-70b-versatile", "qwen2.5:14b"], 
        default="llama-3.3-70b-versatile",
        help="LLM model to use for generating instruction pairs, options: llama-3.3-70b-versatile (groq), qwen2.5:14b (ollama)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for LLM generation"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Ratio of data to use for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all PDF files in the data directory
    pdf_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {args.data_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process all PDF files
    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.data_dir, pdf_file)
        try:
            documents = process_document(
                document_path=pdf_path,
                chunking_strategy=args.chunking_strategy,
                strategy="hi_res"
            )
            all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
        break
    
    logger.info(f"Processed {len(all_documents)} document chunks from {len(pdf_files)} PDFs")
    
    # Generate instruction-response pairs
    pairs = generate_instruction_pairs(
        documents=all_documents,
        llm_model=args.llm_model,
        num_pairs_per_doc=args.num_pairs_per_chunk,
        temperature=args.temperature
    )
    
    # Split the dataset
    train_pairs, val_pairs = split_dataset(pairs, train_ratio=args.train_ratio)
    
    # Save the datasets
    train_path = os.path.join(args.output_dir, f"train.{args.format}")
    val_path = os.path.join(args.output_dir, f"val.{args.format}")
    
    save_dataset(train_pairs, train_path, format=args.format)
    save_dataset(val_pairs, val_path, format=args.format)
    
    # Save metadata
    metadata = {
        "total_examples": len(pairs),
        "train_examples": len(train_pairs),
        "val_examples": len(val_pairs),
        "source_pdfs": pdf_files,
        "chunking_strategy": args.chunking_strategy,
        "llm_model": args.llm_model,
        "temperature": args.temperature,
        "train_ratio": args.train_ratio,
        "format": args.format
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.success(f"Dataset preparation complete! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
