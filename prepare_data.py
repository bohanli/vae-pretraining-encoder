import argparse
import requests
import tarfile
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data downloading")
    parser.add_argument('--dataset', choices=["yahoo", "snli", "short_yelp", "all"], 
        default="all", help='dataset to use')

    args = parser.parse_args()

    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    os.chdir("datasets")

    yahoo_id = "13azGlTuGdzWLCmgDmQPmvb_jcexVWX7i"
    snli_id = "11NHEPxV7OrqmODozxQGezSU8093iUlUx"
    short_yelp_id = "18h8UYr801qr-USCYRzZySi_DSbUkHW71"

    if args.dataset == "yahoo":
        file_id = [yahoo_id]
    elif args.dataset == "snli":
        file_id = [snli_id]
    elif args.dataset == "short_yelp":
        file_id = [short_yelp_id]
    else:
        file_id = [yahoo_id, snli_id, short_yelp_id]

    destination = "datasets.tar.gz"

    for file_id_e in file_id:
        download_file_from_google_drive(file_id_e, destination)  
        tar = tarfile.open(destination, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(destination)

    os.chdir("../")

