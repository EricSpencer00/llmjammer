"""
Example machine learning model script to demonstrate LLMJammer obfuscation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        features, labels (tuple): The feature matrix and label vector
    """
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels


def preprocess_features(features):
    """
    Preprocess the feature matrix.
    
    Args:
        features: Raw feature matrix
        
    Returns:
        Preprocessed feature matrix
    """
    # Normalize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std


class ModelTrainer:
    """
    A class to train and evaluate machine learning models.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
        """
        self.model_type = model_type
        self.model = None
        
    def train_model(self, features, labels, test_size=0.2, random_state=42):
        """
        Train a machine learning model.
        
        Args:
            features: Feature matrix
            labels: Label vector
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            accuracy: The model's accuracy on the test set
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state
        )
        
        # Preprocess features
        X_train = preprocess_features(X_train)
        X_test = preprocess_features(X_test)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            self.model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


def main():
    """Main function to demonstrate the model training process."""
    # This is just a demonstration, so we'll create some dummy data
    np.random.seed(42)
    features = np.random.rand(1000, 10)
    labels = np.random.randint(0, 2, 1000)
    
    # Train a model
    trainer = ModelTrainer()
    accuracy = trainer.train_model(features, labels)
    
    print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
