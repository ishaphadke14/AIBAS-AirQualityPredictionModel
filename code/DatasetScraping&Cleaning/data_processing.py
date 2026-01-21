import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class AQIDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_raw_data(self, filepath='data/raw/aqi_beautifulsoup_scraped.csv'):
        """Load BeautifulSoup scraped data"""
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def load_beautifulsoup_data(self, filepath='data/raw/aqi_beautifulsoup_scraped.csv'):
        """Load data scraped with BeautifulSoup"""
        df = pd.read_csv(filepath)
        
        # Check if timestamp column exists (from BeautifulSoup)
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'date'})
            df['date'] = pd.to_datetime(df['date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Ensure all required columns exist
        required_columns = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'aqi', 'city']
        
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: {col} not in data, adding with default values")
                if col == 'aqi':
                    # Calculate AQI from pollutants if not present
                    if 'pm25' in df.columns:
                        df['aqi'] = df['pm25'] * 2  # Simplified calculation
                    else:
                        df['aqi'] = 50
                elif col == 'city':
                    df['city'] = 'Unknown'
                else:
                    df[col] = 0
        
        return df
    
    def pivot_data(self, df):
        """Convert long format to wide format (if needed)"""
        # Check if data is in long format (parameter column exists)
        if 'parameter' in df.columns and 'value' in df.columns:
            # Pivot to wide format
            df_pivot = df.pivot_table(
                index=['date', 'city'],
                columns='parameter',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # Flatten column names
            df_pivot.columns = [f'{col[1]}' if col[1] else col[0] for col in df_pivot.columns]
        else:
            # Data is already in wide format
            df_pivot = df.copy()
        
        # Calculate AQI if not present
        if 'aqi' not in df_pivot.columns:
            def calculate_aqi(row):
                # Simplified AQI calculation based on PM2.5
                pm25 = row.get('pm25', 0)
                if pd.isna(pm25):
                    return 50  # Default moderate AQI
                
                # AQI calculation based on US EPA standards
                if pm25 <= 12:
                    return (pm25 / 12) * 50
                elif pm25 <= 35.4:
                    return 50 + ((pm25 - 12.1) / (35.4 - 12.1)) * 50
                elif pm25 <= 55.4:
                    return 100 + ((pm25 - 35.5) / (55.4 - 35.5)) * 50
                elif pm25 <= 150.4:
                    return 150 + ((pm25 - 55.5) / (150.4 - 55.5)) * 100
                else:
                    return 300 + ((pm25 - 150.5) / (500.4 - 150.5)) * 200
            
            df_pivot['aqi'] = df_pivot.apply(calculate_aqi, axis=1)
        
        return df_pivot
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Forward fill for time series
        df = df.sort_values(['city', 'date'])
        
        # Group by city and fill
        df_filled = df.groupby('city', group_keys=False).apply(
            lambda x: x.ffill().bfill()
        )
        
        # If still NaN, fill with column mean
        for column in df_filled.select_dtypes(include=[np.number]).columns:
            df_filled[column] = df_filled[column].fillna(df_filled[column].mean())
        
        # Drop any remaining rows with NaN
        df_filled = df_filled.dropna()
        
        return df_filled
    
    def detect_outliers_iqr(self, df, column, threshold=1.5):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return outliers
    
    def remove_outliers(self, df, columns=None):
        """Remove outliers algorithmically"""
        if columns is None:
            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'day_of_week', 'is_weekend']]
        
        clean_df = df.copy()
        
        for column in columns:
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing (for time series)
                clean_df[column] = clean_df[column].clip(lower_bound, upper_bound)
        
        return clean_df
    
    def add_features(self, df):
        """Add temporal features"""
        df = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add lag features for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        lag_columns = [col for col in numeric_cols if col not in ['year', 'month', 'day', 'day_of_week', 'is_weekend']]
        
        for col in lag_columns[:3]:  # Add lags for first 3 numeric columns
            for lag in [1, 2, 3, 7]:
                if f'{col}_lag_{lag}' not in df.columns:  # Avoid duplicate columns
                    df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
        
        # Add rolling statistics for main pollutants
        if 'pm25' in df.columns:
            df['pm25_rolling_mean_7'] = df.groupby('city')['pm25'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        
        if 'aqi' in df.columns:
            df['aqi_rolling_mean_7'] = df.groupby('city')['aqi'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        
        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        # One-hot encode city
        if 'city' in df_encoded.columns:
            city_dummies = pd.get_dummies(df_encoded['city'], prefix='city')
            df_encoded = pd.concat([df_encoded, city_dummies], axis=1)
        
        return df_encoded
    
    def normalize_data(self, df, columns_to_normalize=None):
        """Normalize data algorithmically"""
        if columns_to_normalize is None:
            # Get numeric columns excluding date and encoded columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns_to_normalize = [col for col in numeric_cols 
                                  if not col.startswith('city_') 
                                  and col not in ['year', 'month', 'day', 'day_of_week', 'is_weekend']]
        
        df_normalized = df.copy()
        
        # Use MinMax normalization for better neural network performance
        scaler = MinMaxScaler()
        
        # Normalize each column
        normalized_data = {}
        for col in columns_to_normalize:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Reshape for scaler
                col_data = df[[col]].values
                normalized = scaler.fit_transform(col_data)
                normalized_data[col] = normalized.flatten()
        
        # Update DataFrame with normalized values
        for col, normalized_values in normalized_data.items():
            df_normalized[col] = normalized_values
        
        return df_normalized, scaler
    
    def prepare_for_training(self, df, target_column='aqi'):
        """Prepare features and target for training"""
        # Drop non-feature columns
        columns_to_drop = ['date']
        if 'city' in df.columns:
            columns_to_drop.append('city')
        
        # Keep only numeric columns for features
        features = df.select_dtypes(include=[np.number])
        
        # Remove target from features if present
        if target_column in features.columns:
            features = features.drop(target_column, axis=1)
        
        # Remove any other non-feature columns
        for col in columns_to_drop:
            if col in features.columns:
                features = features.drop(col, axis=1)
        
        # Handle any remaining NaN values
        features = features.fillna(features.mean())
        
        # Get target
        if target_column in df.columns:
            target = df[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        return features, target
    
    def split_data(self, features, target, test_size=0.2):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42, shuffle=True
        )
        
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, source='beautifulsoup'):
        """Complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        if source == 'beautifulsoup':
            print("Loading BeautifulSoup scraped data...")
            df_raw = self.load_beautifulsoup_data()
        else:
            print("Loading raw data...")
            df_raw = self.load_raw_data()
        
        print(f"Raw data shape: {df_raw.shape}")
        
        print("Pivoting data...")
        df_pivot = self.pivot_data(df_raw)
        print(f"After pivoting: {df_pivot.shape}")
        
        print("Handling missing values...")
        df_clean = self.handle_missing_values(df_pivot)
        print(f"After cleaning: {df_clean.shape}")
        
        print("Removing outliers...")
        df_no_outliers = self.remove_outliers(df_clean)
        print(f"After outlier removal: {df_no_outliers.shape}")
        
        print("Adding features...")
        df_features = self.add_features(df_no_outliers)
        print(f"After feature engineering: {df_features.shape}")
        
        print("Encoding categorical variables...")
        df_encoded = self.encode_categorical(df_features)
        print(f"After encoding: {df_encoded.shape}")
        
        print("Normalizing data...")
        df_normalized, scaler = self.normalize_data(df_encoded)
        print(f"After normalization: {df_normalized.shape}")
        
        print("Preparing for training...")
        features, target = self.prepare_for_training(df_normalized)
        print(f"Features shape: {features.shape}, Target shape: {target.shape}")
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(features, target)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Create joint dataset
        df_joint = df_normalized.copy()
        
        # Save all datasets
        os.makedirs('data/processed', exist_ok=True)
        
        df_joint.to_csv('data/processed/joint_data_collection.csv', index=False)
        print(f"✓ Saved joint_data_collection.csv with {len(df_joint)} records")
        
        # Save training data
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.to_csv('data/processed/training_data.csv', index=False)
        print(f"✓ Saved training_data.csv with {len(train_data)} records (80%)")
        
        # Save test data
        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv('data/processed/test_data.csv', index=False)
        print(f"✓ Saved test_data.csv with {len(test_data)} records (20%)")
        
        # Save one activation data entry
        activation_sample = test_data.iloc[[0]]
        activation_sample.to_csv('data/processed/activation_data.csv', index=False)
        print(f"✓ Saved activation_data.csv with 1 record")
        
        print(f"\nData processing complete!")
        print(f"Total records: {len(df_joint)}")
        print(f"Training set: {len(train_data)} records")
        print(f"Test set: {len(test_data)} records")
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'df_joint': df_joint, 'scaler': scaler,
            'feature_names': list(features.columns)
        }

def main():
    """Main function to run data processing"""
    processor = AQIDataProcessor()
    
    print("="*60)
    print("AQI DATA PROCESSING")
    print("="*60)
    
    try:
        # Try to process BeautifulSoup data first
        data_dict = processor.process_pipeline(source='beautifulsoup')
    except FileNotFoundError:
        print("BeautifulSoup data not found, trying default data...")
        data_dict = processor.process_pipeline(source='default')
    
    return data_dict

if __name__ == "__main__":
    main()