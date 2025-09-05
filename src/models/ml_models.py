"""
Các mô hình Machine Learning cho phân loại sử dụng đất/lớp phủ bề mặt đất
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import joblib
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

class RandomForestModel:
    """Random Forest classifier cho phân loại sử dụng đất"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo Random Forest model
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        rf_config = self.config['machine_learning']['random_forest']
        self.model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            random_state=rf_config['random_state'],
            n_jobs=-1
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Huấn luyện mô hình Random Forest
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict: Thông tin training
        """
        self.logger.info("Bắt đầu huấn luyện Random Forest...")
        
        # Chia dữ liệu train/test
        test_size = self.config['model_evaluation']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Huấn luyện mô hình
        self.model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = self.model.predict(X_test)
        
        # Đánh giá
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # Cross validation
        cv_folds = self.config['model_evaluation']['cross_validation_folds']
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
        
        results = {
            'accuracy': accuracy,
            'kappa': kappa,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': self.model.feature_importances_
        }
        
        self.logger.info(f"Accuracy: {accuracy:.3f}, Kappa: {kappa:.3f}")
        self.logger.info(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def save_model(self, path: str):
        """Lưu mô hình"""
        joblib.dump(self.model, path)
        self.logger.info(f"Đã lưu Random Forest model tại: {path}")
    
    def load_model(self, path: str):
        """Tải mô hình"""
        self.model = joblib.load(path)
        self.logger.info(f"Đã tải Random Forest model từ: {path}")

class SVMModel:
    """Support Vector Machine classifier cho phân loại sử dụng đất"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo SVM model
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
        svm_config = self.config['machine_learning']['svm']
        self.model = SVC(
            kernel=svm_config['kernel'],
            C=svm_config['C'],
            gamma=svm_config['gamma'],
            random_state=svm_config['random_state'],
            probability=True
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Huấn luyện mô hình SVM
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict: Thông tin training
        """
        self.logger.info("Bắt đầu huấn luyện SVM...")
        
        # Chia dữ liệu train/test
        test_size = self.config['model_evaluation']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Huấn luyện mô hình
        self.model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = self.model.predict(X_test)
        
        # Đánh giá
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # Cross validation
        cv_folds = self.config['model_evaluation']['cross_validation_folds']
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
        
        results = {
            'accuracy': accuracy,
            'kappa': kappa,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.logger.info(f"Accuracy: {accuracy:.3f}, Kappa: {kappa:.3f}")
        self.logger.info(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Tối ưu hyperparameters cho SVM
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dict: Best parameters và scores
        """
        self.logger.info("Bắt đầu tối ưu hyperparameters cho SVM...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        grid_search = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Cập nhật mô hình với best parameters
        self.model = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán xác suất"""
        return self.model.predict_proba(X)
    
    def save_model(self, path: str):
        """Lưu mô hình"""
        joblib.dump(self.model, path)
        self.logger.info(f"Đã lưu SVM model tại: {path}")
    
    def load_model(self, path: str):
        """Tải mô hình"""
        self.model = joblib.load(path)
        self.logger.info(f"Đã tải SVM model từ: {path}")

class CNNModel:
    """Convolutional Neural Network cho phân loại sử dụng đất"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo CNN model
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.model = None
        self.history = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def build_model(self, num_classes: int) -> keras.Model:
        """
        Xây dựng kiến trúc CNN
        
        Args:
            num_classes: Số lượng lớp
            
        Returns:
            keras.Model: CNN model
        """
        cnn_config = self.config['machine_learning']['cnn']
        input_shape = cnn_config['input_shape']
        
        model = keras.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cnn_config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info("Đã xây dựng CNN model")
        self.logger.info(f"Model summary:\n{model.summary()}")
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Huấn luyện mô hình CNN
        
        Args:
            X: Features (samples, height, width, channels)
            y: Labels
            
        Returns:
            Dict: Thông tin training
        """
        if self.model is None:
            num_classes = len(np.unique(y))
            self.build_model(num_classes)
        
        self.logger.info("Bắt đầu huấn luyện CNN...")
        
        cnn_config = self.config['machine_learning']['cnn']
        
        # Chia dữ liệu train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1
        )
        
        # Huấn luyện
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=cnn_config['batch_size']),
            epochs=cnn_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Đánh giá trên test set
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'kappa': kappa,
            'history': self.history.history,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.logger.info(f"Test Accuracy: {test_accuracy:.3f}")
        self.logger.info(f"Test Loss: {test_loss:.3f}")
        self.logger.info(f"Kappa: {kappa:.3f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted labels
        """
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        return self.model.predict(X)
    
    def save_model(self, path: str):
        """Lưu mô hình"""
        self.model.save(path)
        self.logger.info(f"Đã lưu CNN model tại: {path}")
    
    def load_model(self, path: str):
        """Tải mô hình"""
        self.model = keras.models.load_model(path)
        self.logger.info(f"Đã tải CNN model từ: {path}")

class ModelEvaluator:
    """Class đánh giá và so sánh các mô hình"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo evaluator
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> Dict:
        """Đọc file cấu hình"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """Thiết lập logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def compare_models(self, 
                      models: Dict,
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        So sánh hiệu suất các mô hình
        
        Args:
            models: Dictionary các mô hình
            X_test: Test features
            y_test: Test labels
            
        Returns:
            pd.DataFrame: Bảng so sánh kết quả
        """
        results = []
        
        for model_name, model in models.items():
            self.logger.info(f"Đánh giá mô hình {model_name}...")
            
            # Dự đoán
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                y_pred = np.argmax(model.predict(X_test), axis=1)
            
            # Tính toán metrics
            accuracy = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            
            # Precision, Recall, F1 cho từng lớp
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Kappa': kappa,
                'Precision (macro)': report['macro avg']['precision'],
                'Recall (macro)': report['macro avg']['recall'],
                'F1-score (macro)': report['macro avg']['f1-score']
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        self.logger.info("Kết quả so sánh mô hình:")
        self.logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
    
    def calculate_detailed_metrics(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 class_names: List[str] = None) -> Dict:
        """
        Tính toán metrics chi tiết
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Tên các lớp
            
        Returns:
            Dict: Metrics chi tiết
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Producer's accuracy (Recall) và User's accuracy (Precision)
        producers_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        users_accuracy = np.diag(cm) / np.sum(cm, axis=0)
        
        metrics = {
            'overall_accuracy': accuracy,
            'kappa_coefficient': kappa,
            'classification_report': report,
            'confusion_matrix': cm,
            'producers_accuracy': producers_accuracy,
            'users_accuracy': users_accuracy
        }
        
        return metrics
