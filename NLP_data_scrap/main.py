from newspaper import Article
import streamlit as st
from pathlib import Path
import base64

# Function to scrape articles and save the output
def scrape_articles(urls, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for url in urls:
            article = Article(url)
            article.download()
            article.parse()

            # Get the whole article text
            article_text = article.text

            # Write the article heading and text to the output file
            output_file.write("Article Heading: {}\n".format(article.title))
            output_file.write("Article Text:\n{}\n".format(article_text))
            output_file.write("\n")


# Streamlit app
def main():
    # Title and description
    st.title("Article Scraper")
    st.markdown("This app scrapes articles from URLs and generates an output text file.")

    # Developer name and LinkedIn link
    st.sidebar.markdown("**Developed by:**")
    st.sidebar.markdown("[Rahul](https://www.linkedin.com/in/rahul-dahiya-466323165/)")
    st.sidebar.markdown("**Mentor:**")
    st.sidebar.markdown("[Dr. Anoop V. S.](https://www.linkedin.com/in/anoopvs/)")

    # Option to enter single URL
    st.markdown("### Enter Single URL")
    single_url = st.text_input("Enter a URL")
    if st.button("Scrape Single URL"):
        if single_url:
            urls = [single_url]
            output_file_path = "output_single.txt"
            scrape_articles(urls, output_file_path)
            st.markdown(get_binary_file_downloader_html(output_file_path, 'Download Single URL Output'), unsafe_allow_html=True)
    st.markdown("### OR")
    # Input file upload
    st.markdown("### Upload URLs Text File")
    file = st.file_uploader("Choose a text file", type=["txt"])
    if file is not None:
        # Read the uploaded file
        content = file.read().decode("utf-8")
        urls = content.splitlines()

        # Scrape the articles and save the output
        output_file_path = "output.txt"
        scrape_articles(urls, output_file_path)
        st.markdown(get_binary_file_downloader_html(output_file_path, 'Download Multiple URLs Output'), unsafe_allow_html=True)

# Function to generate a download link for a file
def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_path}">{file_label}</a>'
    return href

if __name__ == "__main__":
    main()
