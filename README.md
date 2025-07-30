# 🐱 Cat vs. Not-Cat: A Deep Dive into Transfer Learning with Keras

Welcome to my project! 🚀 This repository contains a Jupyter Notebook where I explore the power of **transfer learning** to build an image classifier. The goal is to train a model that can distinguish images of **cats** from other objects in the famous **CIFAR-10 dataset**.

This project was a fantastic learning experience, showcasing the entire pipeline from data preparation to model fine-tuning and evaluation.

---

### ✨ Key Features

* **🧠 Transfer Learning:** Leverages the powerful, pre-trained **Xception model** as a base to learn complex features quickly.
* **🎯 Binary Classification:** The model is adapted to solve a specific problem: Is this a cat (output `1`) or not a cat (output `0`)?
* **✂️ Fine-Tuning:** Instead of training the entire network from scratch, only the final layers of the Xception model are "unfrozen" and trained on our specific dataset. This saves a huge amount of time and computational resources.
* **🖼️ Dataset:** Built upon the **CIFAR-10 dataset**, with labels transformed for our binary task.
* **⚙️ Correct Preprocessing:** Utilizes the specific `preprocess_input` function tailored for the Xception model, which is a critical step for achieving good performance.

---

### 🛠️ Tech Stack

* ✅ Python
* ✅ TensorFlow & Keras
* ✅ NumPy
* ✅ Matplotlib
* ✅ Jupyter Notebook (developed in Google Colab)

---

### 🚀 How to Get Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  **Open the notebook:**
    Open `priject_1.ipynb` in a Jupyter environment like Google Colab or a local Jupyter Lab.

3.  **Run the cells:**
    Execute the cells sequentially to load the data, build the model, train it, and see the predictions! ▶️

---

### 겪 Our Learning Journey & Challenges

One of the key challenges in this project was understanding the critical importance of **consistent data preprocessing**.

Initially, the model's performance was low, with predictions for "cat" images being very close to zero (e.g., `0.03`). We discovered a mismatch: the training data was being processed differently than the pre-trained Xception model expected.

**The Fix:** We switched from a simple normalization (`/ 255.0`) to using the official `tensorflow.keras.applications.xception.preprocess_input` function for both the training and testing data. This was a fantastic "Aha!" moment 💡 that significantly improved the model's predictions and taught a valuable lesson: **A model is only as good as the data you feed it!**

---

### 📈 Results & Observations

The final model successfully learns to distinguish between the two classes! As seen in our experiments, it:
* Correctly identifies **'not-cat'** images with high confidence (scores very close to `0.0`).
* Correctly identifies **most 'cat'** images with scores greater than `0.5` (e.g., `0.8`).
* Still finds some images to be "hard cases," which is expected. This highlights that model performance isn't just about correct code, but also about the quantity and quality of training.

---

### 🔮 Future Improvements

This project sets the foundation for many potential improvements:
* **Train for more epochs:** Allow the model more time to learn, using the `EarlyStopping` callback to prevent overfitting.
* **Use more data:** Train on a larger subset of the 50,000 available images.
* **Deeper Fine-Tuning:** Experiment with unfreezing more layers from the Xception base.
* **Tune the Learning Rate:** Use a smaller learning rate for more precise adjustments during the fine-tuning phase.

---

### 📫 Contact Me

Feel free to reach out if you have any questions or suggestions!

* **GitHub:** [your-github-profile-link]
* **LinkedIn:** [your-linkedin-profile-link]
