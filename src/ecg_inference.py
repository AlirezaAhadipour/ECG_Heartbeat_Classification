{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPVrTOQ2PZJGftGVast6tS8"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NZbCWmnUIE_7"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from tensorflow.keras.models import load_model\n",
        "\n",
        "# Function for model inference\n",
        "def predict_ecg_segment(ecg_segment, model_path='models/cnn_base.h5'):\n",
        "    \"\"\"\n",
        "    Takes a single ECG segment and returns the predicted class\n",
        "    \"\"\"\n",
        "    model = load_model(model_path)   # load the trained model\n",
        "\n",
        "    ecg_segment = ecg_segment[np.where(ecg_segment != 0)[0]]   # Remove padding\n",
        "    ecg_segment = np.array(ecg_segment).reshape(1, -1, 1)   # reshape\n",
        "\n",
        "    # prediction\n",
        "    y_pred_prob = model.predict(ecg_segment)\n",
        "    y_pred_class = np.argmax(y_pred_prob, axis=1)[0]   # multi-class classification\n",
        "\n",
        "    class_mapping = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}\n",
        "\n",
        "    return class_mapping[y_pred_class]"
      ]
    }
  ]
}