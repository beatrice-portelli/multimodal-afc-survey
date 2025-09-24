# Survey: Multimodal Automatic Fact-Checking Datasets

This repository contains the survey **"An In-Depth Survey on Multimodal Automatic Fact-Checking Datasets"** (file `survey.pdf`) along with instructions for downloading all the datasets discussed.

## Repository Structure

* `survey.pdf` – PDF of the complete survey [currently under review]
* `README.md` – this document
* `datasets/` – recommended directory for downloading datasets

## Prerequisites

Ensure you have the following tools installed:

* Git
* Python 3.x (optional, for additional processing)
* [gdown](https://github.com/wkentaro/gdown) (`pip install gdown`)

## Downloading the Datasets

It is recommended to run these commands from within the `datasets/` directory.

### FakeClaim

```bash
 git clone https://github.com/Gautamshahi/FakeClaim.git
```

### WarClaim

```bash
 git clone https://github.com/Gautamshahi/WarClaim.git
```

### FineFake

```bash
 gdown https://drive.google.com/uc?id=16D9ix7ZOisa4VVBznBTBcv1N7TA-jodH
```
Only the video links are provided; retrieve the information from YouTube using whichever method you prefer.

### MOCHEG

1. Complete the questionnaire at:

   ```
   https://docs.google.com/forms/d/e/1FAIpQLScAGehM6X9ARZWW3Fgt7fWMhc_Cec6iiAAN4Rn1BHAk6KOfbw/viewform
   ```
2. Copy the link you receive and run:

   ```bash
   wget <received_link>       # e.g., mocheg_with_tweet_2023_03.tar.gz
   tar -xvzf mocheg_with_tweet_2023_03.tar.gz
   ```

### r/Fakeddit

```bash
 # Download the main folder
 gdown https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm --folder
 cd "Fakeddit datasetv2.0/"
 # Public images
 gdown https://drive.google.com/uc?id=1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b
 tar -xvjf public_images.tar.bz2
 # Comments
 gdown https://drive.google.com/uc?id=1hjcLrwMOGJBHrB6bqlaIhn6N09X0lr85
 unzip all_comments.tsv.zip
```

### Fauxtography

```bash
 git clone https://gitlab.com/didizlatkova/fake-image-detection
```

### CovID I

```bash
 gdown "https://drive.google.com/uc?id=1bjMrvPIgwAXt_nvtmP0vFqEqEtYq_YmS"
```

### CovID II

```bash
 gdown "https://drive.google.com/uc?id=1ivBi9T0GoY3vkQiabWEQg6CnPSvkpAh7"
```

### Evons

```bash
 # Download the CSV directly from Dropbox
 wget "https://www.dropbox.com/scl/fi/k05g7rr5wiqccay7xope6/evons.csv?rlkey=9riao2g3uz3aiktijfgaheplf&dl=1" -O evons.csv
```

### ReCOVery

```bash
 git clone https://github.com/apurvamulay/ReCOVery.git
```

### Factify2

1. Register for the competition:

   ```
   https://codalab.lisn.upsaclay.fr/competitions/8275
   ```
2. After registering, navigate to the **Participate > Files** tab and download the zipped dataset.
3. Complete the form to receive passwords for the ZIP files:

   ```
   https://docs.google.com/forms/d/e/1FAIpQLSfTmTUsr0LSjdvVwlGmD7a1Ek9ytIzCN8pIew1Hym0AavTbZg/viewform
   ```
4. Unzip the downloaded archive using the provided password:

   ```bash
   unzip Factify2.zip
   ```

> Images are provided as links within the dataset. For any missing images, please contact: **[defactifyaaai@gmail.com](mailto:defactifyaaai@gmail.com)**

## Organizing the Data

We recommend keeping each dataset in its own subdirectory within `datasets/`. For example:

```
 datasets/
 ├── FakeClaim/
 ├── WarClaim/
 ├── FineFake/
 ├── MOCHEG/
 ├── Fakeddit datasetv2.0/
 ├── fake-image-detection/    # Fauxtography
 ├── CovID_I/
 ├── CovID_II/
 ├── evons.csv
 ├── Factify2/
 └── ReCOVery/
```

---

For detailed dataset analyses, please refer to `survey.pdf`.
