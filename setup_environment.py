import subprocess

def run_command(command):
    """Run a system command and handle errors."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)

# Install required packages globally
print("Installing required packages globally...")
run_command('pip install nltk spacy scikit-learn')

# Download NLTK data
print("Downloading NLTK data...")
run_command('python -m nltk.downloader punkt stopwords')

# Download SpaCy model
print("Downloading SpaCy model...")
run_command('python -m spacy download en_core_web_sm')

print("Setup complete. Required packages are installed globally.")
