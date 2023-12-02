import base64
import requests
import boto3
import boto3
import openai
import json
 
responses = []
 
# Connect to S3  
s3 = boto3.client('s3', aws_access_key_id='AKIATCBWSKWQW54UNGR2', aws_secret_access_key='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')
bucket_name = 'admassignment5'
 
# Connect to ChatGPT API
openai.api_key = "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"
 
# List images in S3 bucket
images = s3.list_objects(Bucket=bucket_name)['Contents'] 
 
 
