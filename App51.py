import base64
import requests
import boto3
import boto3
import openai
import json
 
responses = []
 
# Connect to S3  
s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')
bucket_name = 'admassignment5'
 
# Connect to ChatGPT API
openai.api_key = ""
 
# List images in S3 bucket
images = s3.list_objects(Bucket=bucket_name)['Contents'] 

def form_and_table_understanding(image_path, prompt_text,key):
  """ form_and_table_understanding """
  base64_image = image_path # Path to your image
  headers = {"Content-Type": "application/json","Authorization": f"Bearer "}
  payload = {
      "model": "gpt-4-vision-preview",
      #"response_format" : { "type": "json_object" },
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_text
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
 
  return(response.json(),key)
 
 
