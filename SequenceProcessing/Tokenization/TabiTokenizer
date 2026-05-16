import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


# -----------------------------
# Argument Parser
# -----------------------------

parser = argparse.ArgumentParser(
    description="Local tokenization pipeline using Datatrove."
)

parser.add_argument(
    "data_path",
    type=str,
    help="Path to the JSONL data file."
)

parser.add_argument(
    "output_name",
    type=str,
    help="Name of the output run."
)

parser.add_argument(
    "--n_tasks",
    type=int,
    default=4,
    help="Number of parallel tasks."
)

parser.add_argument(
    "--max_toks",
    type=int,
    default=int(1e8),
    help="Maximum tokens per output file."
)

parser.add_argument(
    "--tokenizer",
    type=str,
    default="gpt2",
    help="Tokenizer name from HuggingFace."
)

parser.add_argument(
    "--text_key",
    type=str,
    default="text",
    help="JSON key containing the text."
)

parser.add_argument(
    "--sample",
    type=float,
    default=1.0,
    help="Sampling rate between 0 and 1."
)

parser.add_argument(
    "--jsonl_output",
    "-jo",
    type=str,
    default=None,
    help="Optional path to save sampled JSONL."
)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    args = parser.parse_args()

    # -----------------------------
    # Create local directories
    # -----------------------------

    base_dir = Path(__file__).parent

    tokenized_dir = base_dir / "tokenized" / args.output_name
    tmp_dir = base_dir / "tmp" / args.output_name
    logs_dir = base_dir / "logs" / args.output_name

    tokenized_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Build pipeline
    # -----------------------------

    pipeline = [

        JsonlReader(
            args.data_path,
            text_key=args.text_key,
        ),

        SamplerFilter(
            rate=args.sample
        ),

        *(
            [JsonlWriter(args.jsonl_output)]
            if args.jsonl_output
            else []
        ),

        DocumentTokenizer(

            output_folder=str(tokenized_dir),

            local_working_dir=str(tmp_dir),

            tokenizer_name_or_path=args.tokenizer,

            eos_token=None,

            batch_size=1000,

            max_tokens_per_file=args.max_toks,

            shuffle=True,
        ),
    ]

    # -----------------------------
    # Run locally
    # -----------------------------

    executor = LocalPipelineExecutor(

        pipeline=pipeline,

        tasks=args.n_tasks,
    )

    executor.run()

    print("\nTokenization completed successfully.")
