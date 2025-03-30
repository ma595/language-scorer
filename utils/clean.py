import re
import pandas as pd

# Read raw lines (handle embedded newlines within quotes, and ^M cleanup)
with open("data.csv", "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read().replace("\r", "").replace("^M", "")  # Remove ^M and \r

# Use regex to extract text in quotes followed by numeric values
pattern = r'"(.*?)"\s*((?:\d+\s*)+)'  # Captures the quoted content and trailing numbers
matches = re.findall(pattern, raw, re.DOTALL)

data = []
for text, nums in matches:
    num_list = [int(n) for n in nums.strip().split()]
    score = num_list[-1]  # Only keep the last number
    data.append((text.strip(), score))

# Put into DataFrame
df = pd.DataFrame(data, columns=["content", "score"])

# Save cleaned version
df.to_csv("cleaned_dataset.csv", index=False)
