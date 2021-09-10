{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "fallen.ipynb",
      "private_outputs": true,
      "provenance": [],
      "collapsed_sections": [],
      "machine_shape": "hm",
      "mount_file_id": "1pBniEQtcKYidMdMt1u4Z8gr3juX1tlxF",
      "authorship_tag": "ABX9TyPIWBKTgmhVidI00eVc+Glz",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jeonniga/-/blob/master/custom_trainer.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "OqEOyJHTDdNB"
      },
      "source": [
        "%cd /content\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GvuJ2Cue0xf0"
      },
      "source": [
        "!curl -L \"https://app.roboflow.com/ds/m9dnjx4upk?key=CEYV4fogVj\" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3JsjA7BV5p_D"
      },
      "source": [
        "%cd /content/\n",
        "!git clone https://github.com/ultralytics/yolov5.git"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6-wfJowz7133"
      },
      "source": [
        "%cd /content/yolov5/\n",
        "!pip install -r requirements.txt"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HwqNVngR8jbO"
      },
      "source": [
        "!cat /content/data.yaml"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Q_zpVYeV9CYb"
      },
      "source": [
        "%cd /\n",
        "from glob import glob\n",
        "img_list = glob('/content/train/images/*.jpg')\n",
        "print(len(img_list))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bncLlgtQ90Dz"
      },
      "source": [
        "'''\n",
        "from sklearn.model_selection import train_test_split\n",
        "train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)\n",
        "print(len(train_img_list), len(val_img_list))\n",
        "'''"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xI9iEwrN-mkq"
      },
      "source": [
        "'''\n",
        "with open('/content/fallen_dataset/train.txt','w') as f:\n",
        "  f.write('\\n'.join(train_img_list)+'\\n')\n",
        "\n",
        "with open('/content/fallen_dataset/val.txt', 'w') as f:\n",
        "  f.write('\\n'.join(val_img_list)+'\\n')\n",
        "'''"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r7-Qh9WC_LNY"
      },
      "source": [
        "'''\n",
        "import yaml\n",
        "\n",
        "with open('/content/fallen_dataset/data.yaml', 'r') as f:\n",
        "  data = yaml.load(f)\n",
        "\n",
        "print(data)\n",
        "\n",
        "data['train'] = '/content/fallen_dataset/train.txt'\n",
        "data['val'] = '/content/fallen_datset/val.txt'\n",
        "\n",
        "with open('/content/fallen_dataset/data.yaml','w') as f:\n",
        "  yaml.dump(data,f)\n",
        "\n",
        "print(data)\n",
        "'''"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xuQba2RWAnAd"
      },
      "source": [
        "%cd /content/yolov5/\n",
        "!python train.py --img 416 --batch 16 --epochs 5 --data /content/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name fallendataset"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "npD42H4iGGSr"
      },
      "source": [
        "%load_ext tensorboard \n",
        "%tensorboard --logdir /content/yolov5/runs/"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4beX8nZuqN78"
      },
      "source": [
        "%cd /content/yolov5/"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gfJ6kd7C5v0U"
      },
      "source": [
        "\n",
        "from IPython.display import Image\n",
        "import os\n",
        "val_img_path = '/content/valid/images/'\n",
        "!python detect.py --weights /content/yolov5/runs/train/fallendataset/weights/last.pt  --img 416 --conf 0.5 --source \"{val_img_path}\"\n",
        "Image(os.path.join('/content/yolov5/inference/output'), os.path.basename(val_img_path))\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}