#!/usr/bin/env python3
"""
Verification script for generation module.
Checks code structure, imports, and syntax without requiring all dependencies.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any


class ModuleVerifier:
    """Verify generation module structure and quality."""

    def __init__(self, module_path: str):
        self.module_path = Path(module_path)
        self.results = []

    def check_file_exists(self, filename: str) -> bool:
        """Check if a file exists in the module."""
        filepath = self.module_path / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        size = filepath.stat().st_size if exists else 0
        self.results.append(f"{status} {filename}: {size:,} bytes")
        return exists

    def analyze_file(self, filename: str) -> Dict[str, Any]:
        """Analyze a Python file's structure."""
        filepath = self.module_path / filename

        with open(filepath, 'r') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}

        # Extract information
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Count docstrings
        docstrings = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
            and ast.get_docstring(node)
        )

        # Count type hints
        type_hints = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and (node.returns or any(arg.annotation for arg in node.args.args))
        )

        return {
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "docstrings": docstrings,
            "type_hints": type_hints,
            "lines": len(content.split('\n'))
        }

    def verify_structure(self):
        """Verify overall module structure."""
        print("=" * 70)
        print("Generation Module Verification")
        print("=" * 70)
        print(f"\nModule path: {self.module_path}\n")

        # Check all required files exist
        print("File Existence Check:")
        print("-" * 70)
        required_files = [
            "__init__.py",
            "llm_client.py",
            "prompts.py",
            "validators.py",
            "answer_generator.py",
            "README.md"
        ]

        all_exist = all(self.check_file_exists(f) for f in required_files)
        for result in self.results:
            print(result)

        if not all_exist:
            print("\n✗ Missing required files!")
            return False

        print("\n✓ All required files present\n")
        return True

    def verify_code_quality(self):
        """Verify code quality metrics."""
        print("Code Quality Analysis:")
        print("-" * 70)

        python_files = [
            "llm_client.py",
            "prompts.py",
            "validators.py",
            "answer_generator.py"
        ]

        for filename in python_files:
            print(f"\n{filename}:")
            analysis = self.analyze_file(filename)

            if "error" in analysis:
                print(f"  ✗ {analysis['error']}")
                continue

            print(f"  Lines of code: {analysis['lines']}")
            print(f"  Classes: {len(analysis['classes'])} - {', '.join(analysis['classes'][:3])}")
            print(f"  Functions: {len(analysis['functions'])}")
            print(f"  Docstrings: {analysis['docstrings']}")
            print(f"  Type hints: {analysis['type_hints']}")

            # Quality checks
            if analysis['docstrings'] > 0:
                print("  ✓ Has docstrings")
            else:
                print("  ✗ Missing docstrings")

            if analysis['type_hints'] > 0:
                print("  ✓ Has type hints")
            else:
                print("  ✗ Missing type hints")

    def verify_imports(self):
        """Verify key imports are present."""
        print("\n\nKey Dependencies Check:")
        print("-" * 70)

        expected_imports = {
            "llm_client.py": ["groq", "src.config", "src.utils.logger", "src.utils.exceptions"],
            "prompts.py": [],
            "validators.py": ["src.generation.llm_client", "src.generation.prompts", "src.config"],
            "answer_generator.py": ["src.generation.llm_client", "src.generation.prompts", "src.generation.validators"]
        }

        for filename, expected in expected_imports.items():
            if not expected:
                continue

            analysis = self.analyze_file(filename)
            if "error" in analysis:
                continue

            imports = analysis['imports']
            print(f"\n{filename}:")

            for exp in expected:
                # Check if import or any submodule is imported
                found = any(exp in imp or imp in exp for imp in imports)
                status = "✓" if found else "✗"
                print(f"  {status} {exp}")

    def verify_features(self):
        """Verify key features are implemented."""
        print("\n\nFeature Implementation Check:")
        print("-" * 70)

        features = {
            "llm_client.py": [
                "retry_with_exponential_backoff",
                "LLMClient",
                "generate",
                "LLMResponse"
            ],
            "prompts.py": [
                "PromptTemplate",
                "get_system_prompt",
                "format_context_documents",
                "create_qa_prompt"
            ],
            "validators.py": [
                "CitationValidator",
                "AnswerVerifier",
                "CitationValidationResult",
                "AnswerVerificationResult"
            ],
            "answer_generator.py": [
                "AnswerGenerator",
                "GeneratedAnswer",
                "reorder_context_for_attention",
                "compute_confidence_score"
            ]
        }

        for filename, expected_features in features.items():
            analysis = self.analyze_file(filename)
            if "error" in analysis:
                continue

            all_names = analysis['classes'] + analysis['functions']
            print(f"\n{filename}:")

            for feature in expected_features:
                found = feature in all_names
                status = "✓" if found else "✗"
                print(f"  {status} {feature}")

    def generate_summary(self):
        """Generate summary statistics."""
        print("\n\nSummary:")
        print("=" * 70)

        total_lines = 0
        total_classes = 0
        total_functions = 0
        total_docstrings = 0

        python_files = [
            "llm_client.py",
            "prompts.py",
            "validators.py",
            "answer_generator.py"
        ]

        for filename in python_files:
            analysis = self.analyze_file(filename)
            if "error" not in analysis:
                total_lines += analysis['lines']
                total_classes += len(analysis['classes'])
                total_functions += len(analysis['functions'])
                total_docstrings += analysis['docstrings']

        print(f"Total Lines of Code: {total_lines:,}")
        print(f"Total Classes: {total_classes}")
        print(f"Total Functions/Methods: {total_functions}")
        print(f"Total Docstrings: {total_docstrings}")
        print(f"\nDocumentation Coverage: {(total_docstrings / max(total_classes + total_functions, 1) * 100):.1f}%")

    def run_all_checks(self):
        """Run all verification checks."""
        if not self.verify_structure():
            return False

        self.verify_code_quality()
        self.verify_imports()
        self.verify_features()
        self.generate_summary()

        print("\n" + "=" * 70)
        print("Verification Complete!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set GROQ_API_KEY in .env file")
        print("3. Run integration test: python test_generation_integration.py")
        print("4. See src/generation/README.md for usage examples")
        print("=" * 70)

        return True


def main():
    """Main verification function."""
    # Find generation module
    current_dir = Path(__file__).parent
    generation_path = current_dir / "src" / "generation"

    if not generation_path.exists():
        print(f"Error: Generation module not found at {generation_path}")
        return

    verifier = ModuleVerifier(generation_path)
    verifier.run_all_checks()


if __name__ == "__main__":
    main()
