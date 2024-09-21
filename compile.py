#!/home//hackathon/.venv/bin/python3
import sys
import os
import re
import argparse
import subprocess

prefix = """
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
"""

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Append suffix to input file and execute as Python script.")
parser.add_argument('-i', '--input', type=str, required=True, help='Path to input file')
parser.add_argument('-o', '--output', type=str, required=True, help='Path to output file')
parser.add_argument('-t', '--type', type=str, required=True, help='Compilation type')
args = parser.parse_args()

# Read the input file
try:
    with open(args.input, 'r') as infile:
        input_content = infile.read()
except FileNotFoundError:
    print(f"Error: Input file {args.input} not found.")
    sys.exit(1)

def extract_kernel_info(code):
    # Define regex patterns
    signature_pattern = r'#\s*signature\s*=\s*\"(.*?)\"'
    kernel_name_pattern = r'@triton.jit\s+def\s+(\w+)\('

    # Extract signature
    signature_match = re.search(signature_pattern, code)
    signature = signature_match.group(1) if signature_match else None

    # Extract kernel name
    kernel_name_match = re.search(kernel_name_pattern, code)
    kernel_name = kernel_name_match.group(1) if kernel_name_match else None

    return kernel_name, signature

kernel_name, signature = extract_kernel_info(input_content)

if not kernel_name or not signature:
    print(f"Cannot extract kernel name and signature")
    sys.exit(1)

dump_type = args.type if args.type != 'asm' else 'llir'

# Define the suffix to append
suffix = """
src = triton.compiler.ASTSource(
    fn={0},
    signature="{1}",
)
ret = triton.compile(
    src,
)
print(ret.asm["{2}"])
""".format(kernel_name, signature, dump_type)


# Append the suffix to the input content
modified_content = prefix + input_content + suffix

# Write the modified content to a temporary file
temp_script = 'temp_script.py'
with open(temp_script, 'w') as tempfile:
    tempfile.write(modified_content)

# Define environment variables to pass
env_vars = os.environ.copy()
env_vars['TRITON_SHARED_OPT_PATH'] = '/home//hackathon/triton_shared/triton/python/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt'  # Hardcoded value for var1
env_vars['LLVM_BINARY_DIR'] = '/home//.triton/llvm/llvm-c08c6a71-ubuntu-x64/bin'  # Hardcoded value for var2

# Run the modified script and capture stdout with the specified environment variables
try:
    result = subprocess.run(['/home//hackathon/.venv/bin/python3', temp_script], capture_output=True, text=True, env=env_vars)
    output = result.stdout
except Exception as e:
    print(f"Error running the script: {e}")
    sys.exit(1)

if args.type != 'asm':
    # Write the stdout to the output file
    with open(args.output, 'w') as outfile:
        outfile.write(output)

    # Clean up the temporary script file
    os.remove(temp_script)
    sys.exit(0)

ttshared_script = 'temp.tts'
# Write the stdout to the output file
with open(ttshared_script, 'w') as outfile:
    outfile.write(output)

LLVM_COMPILER = env_vars['LLVM_BINARY_DIR'] + '/llc'
# Run MAIA compiler
try:
    result = subprocess.run([LLVM_COMPILER, ttshared_script, '-o', args.output], capture_output=True, text=True, env=env_vars)
    output = result.stdout
except Exception as e:
    print(f"Error running llc compiler: {e}")
    sys.exit(1)


# Clean up the temporary script file
os.remove(temp_script)
os.remove(ttshared_script)