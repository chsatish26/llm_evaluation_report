"""
Setup and validation script for LLM Evaluation Framework
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'boto3': 'AWS Bedrock support',
        'reportlab': 'PDF report generation',
        'dotenv': 'Configuration management'
    }
    
    optional = {
        'openai': 'OpenAI model support'
    }
    
    missing = []
    print("\nChecking required dependencies:")
    for package, description in required.items():
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description} - MISSING")
            missing.append(package)
    
    print("\nChecking optional dependencies:")
    for package, description in optional.items():
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"○ {package}: {description} - Not installed (optional)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_env_file():
    """Check if .env file exists"""
    if not Path('.env').exists():
        print("\n⚠ .env file not found")
        print("Copy .env.example to .env and configure your settings:")
        print("  cp .env.example .env")
        return False
    
    print("\n✓ .env file found")
    return True


def validate_env_config():
    """Validate environment configuration"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\nValidating configuration:")
    
    issues = []
    
    # Check provider
    provider = os.getenv('PROVIDER', '').lower()
    if provider not in ['aws', 'openai']:
        issues.append("PROVIDER must be 'aws' or 'openai'")
    else:
        print(f"✓ Provider: {provider}")
    
    # Check AWS configuration
    if provider == 'aws' or 'aws' in os.getenv('EVAL_MODEL_MAPPING', '').lower():
        if not os.getenv('AWS_REGION'):
            issues.append("AWS_REGION not set")
        else:
            print(f"✓ AWS Region: {os.getenv('AWS_REGION')}")
        
        if not os.getenv('AWS_ACCESS_KEY_ID'):
            print("⚠ AWS_ACCESS_KEY_ID not set (will use default credentials)")
    
    # Check OpenAI configuration
    if provider == 'openai' or 'openai' in os.getenv('EVAL_MODEL_MAPPING', '').lower():
        if not os.getenv('OPENAI_API_KEY'):
            issues.append("OPENAI_API_KEY not set but OpenAI provider selected")
        else:
            print("✓ OpenAI API key configured")
    
    # Check template file
    template_path = os.getenv('PROMPT_TEMPLATE_PATH', 'prompt_templates_enhanced_deepeval.json')
    if not Path(template_path).exists():
        issues.append(f"Template file not found: {template_path}")
    else:
        print(f"✓ Template file: {template_path}")
    
    # Check output directory
    output_path = os.getenv('OUTPUT_REPORT_PATH', 'evaluation_reports')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_path}")
    
    if issues:
        print("\n❌ Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


def create_directory_structure():
    """Create necessary directories"""
    print("\nCreating directory structure:")
    
    directories = [
        'evaluation_reports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return True


def validate_template_file():
    """Validate template file structure"""
    import json
    
    template_path = os.getenv('PROMPT_TEMPLATE_PATH', 'prompt_templates_enhanced_deepeval.json')
    
    if not Path(template_path).exists():
        print(f"\n⚠ Template file not found: {template_path}")
        return False
    
    print(f"\nValidating template file: {template_path}")
    
    try:
        with open(template_path, 'r') as f:
            templates = json.load(f)
        
        if not isinstance(templates, dict):
            print("❌ Template file must be a JSON object")
            return False
        
        total_tests = sum(len(tests) for tests in templates.values())
        print(f"✓ Found {len(templates)} categories with {total_tests} total tests")
        
        for category, tests in templates.items():
            print(f"  - {category}: {len(tests)} tests")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in template file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading template file: {e}")
        return False


def main():
    """Main setup and validation"""
    print("=" * 70)
    print("LLM Evaluation Framework - Setup & Validation")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Environment Config", validate_env_config),
        ("Directory Structure", create_directory_structure),
        ("Template File", validate_template_file)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"\n❌ Error in {check_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ Setup validation PASSED")
        print("=" * 70)
        print("\nYou can now run the evaluation:")
        print("  python main.py")
    else:
        print("❌ Setup validation FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before running the evaluation.")
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())