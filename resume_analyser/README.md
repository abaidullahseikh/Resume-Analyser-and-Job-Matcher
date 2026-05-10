# Resume Analyser & Job Matcher

This is a project I made for analysing how well a resume matches a job description. You upload both files (or paste the text), click a button, and it gives you a score out of 100 with a breakdown of what matched and what didn't.

It doesn't use any AI like ChatGPT — it's all regex and rule-based NLP with a small matching model. Everything shown on the dashboard is pulled directly from your resume text, nothing is made up.

---

## Before You Start

Make sure you have:

- Python 3.10 or newer
- pip
- Around 1.5 GB of free disk space (a model gets downloaded the first time you run it)

---

## How to Run It

### 1. Download the project

```bash
git clone <repo-url>
cd resume_analyser
```

### 2. Create a virtual environment

This keeps the packages separate from your system Python. I'd recommend doing this.

**Windows:**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```bash
pip install flask==3.0.0 nltk==3.9.4 pdfplumber==0.11.9 PyPDF2==3.0.1 \
    pdfminer.six==20251230 textstat==0.7.13 scikit-learn==1.8.0 \
    networkx==3.6.1 pyphen==0.17.2 pillow==12.2.0 \
    sentence-transformers==3.2.1 numpy
```

Or if there's a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one time only)

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Start the server

Make sure you're inside the `resume_analyser/` folder when you run this:

```bash
python app.py
```

Then open your browser and go to **http://127.0.0.1:5000**

> **Note:** The first time you run an analysis it will download a ~90 MB model from HuggingFace. This can take a minute. After that it's saved locally and runs fast.

---

## Using the App

1. Go to `http://127.0.0.1:5000`
2. Paste your resume text or upload a PDF
3. Paste the job description or upload a PDF
4. Click **Analyse**
5. The dashboard will show your results



---


### Make sure your package versions match

The scores depend on the exact versions listed in step 3. Using different versions (especially `sentence-transformers` or `scikit-learn`) might give slightly different numbers.

