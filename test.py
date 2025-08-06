import subprocess
import sys

# Get the number of questions from command line argument, default to 5 if not provided
n_questions = int(sys.argv[1])
n_languages = int(sys.argv[2]) if len(sys.argv) > 2 else -1

subprocess.run(["python", "pipeline/generate.py", str(n_questions), str(n_languages)])
subprocess.run(["python", "pipeline/extractGT.py"])
subprocess.run(["python", "pipeline/process.py"])
subprocess.run(["python", "pipeline/judge.py"]) 
subprocess.run(["python", "pipeline/analyze.py"])