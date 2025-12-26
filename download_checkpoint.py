import gdown

file_id = "1oDvaBVKwlUmsAbtOYlFSqapi3ti5YeB3"
url = f"https://drive.google.com/uc?id={file_id}"
output = "checkpoint.pth"
gdown.download(url, output, quiet=False)


# https://drive.google.com/file/d/1oDvaBVKwlUmsAbtOYlFSqapi3ti5YeB3/view?usp=sharing