import re
import os

filepath = r"d:\Projects\College\DART\Project Report\DART_Project_Report.html"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find $$ ... $$ blocks. 
# Using re.DOTALL to match across lines.
# This assumes $$ are used for display math.
def fix_math_block(match):
    block = match.group(0)
    # Replace <em> and </em> with _
    fixed_block = block.replace('<em>', '_').replace('</em>', '_')
    return fixed_block

# Apply fix to all $$ blocks
new_content = re.sub(r'\$\$(.*?)\$\$', fix_math_block, content, flags=re.DOTALL)

# Also check for \[ ... \] blocks if any
new_content = re.sub(r'\\\[(.*?)\\\]', fix_math_block, new_content, flags=re.DOTALL)

# Also check for inline math $ ... $ 
# This is riskier as $ is common currency, but let's check if we can be smart.
# The user issue is specifically with display math J(\pi)
# So we focus on $$ first.

if content != new_content:
    print("Found and fixed math blocks.")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
else:
    print("No changes made.")
