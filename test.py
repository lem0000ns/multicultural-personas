import subprocess

subprocess.run(["python", "pipeline/generate.py"])
subprocess.run(["python", "pipeline/extractGT.py"])
subprocess.run(["python", "pipeline/judge_weave_enhanced.py"]) 
subprocess.run(["python", "pipeline/analyze.py"])