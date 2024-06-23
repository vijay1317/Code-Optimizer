import math

import streamlit as st
import os
import matplotlib.pyplot as plt
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from openai import OpenAI





client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "$$$$$"
)

# Set the NVIDIA API key
os.environ['NVIDIA_API_KEY'] = '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'  # Replace with your actual API key

# Initialize the NVIDIA Chat model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")




def get_time_complexity(code):
    response =''
    prompt = f'''The time complexity of given code is :{code}'''
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            response+= chunk.choices[0].delta.content

    return response


def get_optimized(code):
    response =''
    prompt = f'''Optimize the given code snippet :{code}'''
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )

    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            response+= chunk.choices[0].delta.content

    return response




# Streamlit app layout
st.title("Code Optimizer")

if 'code_input' not in st.session_state:
    st.session_state.code_input = ""
code_input = st.text_area("Code Input",height=300,value=st.session_state.code_input)
col1, col2,col3 = st.columns(3)
if col1.button("Analyze"):
    if code_input:
        with st.spinner("Analyzing the code..."):
            complexity = get_time_complexity(code_input)
        st.success("Optimization complete!")
        st.write(f"**Time Complexity:** {complexity}")

    else:
        st.error("Please enter some code to analyze.")


if(col2.button("Optimize")):
    if code_input:
        with st.spinner("Optimizing the code..."):
            optimized_code = get_optimized(code_input)
        st.success("Analysis complete!")
        st.code(optimized_code)
        # st.write(f"**Time Complexity:** {optimized_code}")

    else:
        st.error("Please enter some code to analyze.")