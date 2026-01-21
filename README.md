# AI-Based E-Commerce Recommendation System 🛍️

A smart e-commerce application built with Streamlit that leverages machine learning to provide personalized product recommendations.

## 🚀 Live Demo
**Check out the live application here:** [AI-Based E-Commerce Recommendation System](https://ai-based-e-commerce-recommendation-system-revanth.streamlit.app/)

## 🌟 Features

- **Interactive User Interface**: Built with Streamlit for a smooth and responsive experience.
- **Product Discovery**:
  - **Search**: Find products by name with hybrid fallback recommendations.
  - **Filters**: Filter products by Brand and Rating.
  - **Categories**: Browse by categories like Nail Polish, Skin Care, Hair Care, etc.
- **Smart Recommendations**:
  - **Content-Based Filtering**: Suggests items similar to what you are viewing button.
  - **Collaborative Filtering**: Personalized "Recommended for You" section based on user history.
  - **Item-Based Collaborative Filtering**: "Users Also Bought" suggestions.
  - **Hybrid Approach**: Combines methods for better accuracy, especially when search fails.
  - **Top Rated**: Showcases highest-rated products for new users.
- **User Simulation**:
  - **User ID Login**: Simulate different users by entering a User ID.
  - **History Tracking**: Tracks "Previously Rated" items for returning users.
- **Shopping Cart**:
  - Add items to cart.
  - View cart summary with tax calculation.
  - Simulated checkout process.

## 🛠️ Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

To run the application locally:

```bash
streamlit run demo_streamlit.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

## 📂 Project Structure

- `demo_streamlit.py`: Main application file containing the UI and logic.
- `clean_data.csv`: Dataset used for products and ratings.
- `preprocess_data.py`: Data cleaning and processing scripts.
- `collaborative_based_filtering.py`: User-based recommendation logic.
- `content_based_filtering.py`: Content-based recommendation logic.
- `item_based_collaborative_filtering.py`: Item-item recommendation logic.
- `hybrid_approach.py`: Hybrid recommendation logic.
- `evaluation_metrics.py`: Metrics for evaluating recommendation models.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
