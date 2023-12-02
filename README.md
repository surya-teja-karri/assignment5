# assignment5

Image Analysis Streamlit Application

Overview
This Streamlit application provides two main features: Image Match and Image Search by Text. It utilizes advanced technologies like TensorFlow, OpenAI's GPT-3, and SerpApi for image processing and text-based image search functionalities.

Image Match
This feature allows users to upload an image and find similar images based on style. The application uses a pre-trained VGG19 model from TensorFlow for style extraction and comparison.

Image Search by Text
In this feature, users can input a text query, and the application will use GPT-3 to understand and expand on the query. It then searches for relevant images using SerpApi's Google Image Search functionality.

Installation and Setup
Clone the repository and navigate into it.
Install required Python packages.
Set up API keys for OpenAI and SerpApi.
To run the application: streamlit run app.py

Usage
Image Match: Upload an image to find similar images based on style.
Image Search by Text: Enter a text query to find relevant images.

Privacy Requirements :
Removed AWS , Sceret key , API keys , chatgpt openAI key and snowflake credentials for privacy purpose. Please use your own while running the code .

Codelab:
https://codelabs-preview.appspot.com/?file_id=10roBrOUIhijRykfH6x-5EtOhnhudcGFgadANVnVZN7M#0
