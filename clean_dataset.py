import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
from collections import Counter
from sklearn.impute import KNNImputer
import nltk
from nltk.metrics import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# Download NLTK resources
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Load existing spelling corrections if available
try:
    import json
    import os
    
    CORRECTIONS_FILE = 'spelling_corrections.json'
    
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, 'r') as f:
            saved_corrections = json.load(f)
            print(f"Loaded {len(saved_corrections)} existing spelling corrections")
    else:
        saved_corrections = {}
        print("No existing corrections file found. Will create one after processing.")
except Exception as e:
    print(f"Error loading spelling corrections: {e}")
    saved_corrections = {}

# Load the corrupt dataset
df = pd.read_csv('corrupt_fictional_dataset.csv')

# Expanded dictionary for common event name corrections
event_name_corrections = {
    # Core event type corrections - these are essential
    'sreakthrough': 'Breakthrough',
    'Brakhrugh': 'Breakthrough', 
    'Breakt': 'Breakthrough',
    '$r*akthrough': 'Breakthrough',
    'Br%?kthrough': 'Breakthrough',
    'Breakthrsugh': 'Breakthrough',
    
    'Discov%ry': 'Discovery',
    'Disc@very': 'Discovery',
    'Dscovery': 'Discovery',
    'isovery': 'Discovery',
    
    'Re$@rm': 'Reform',
    'Re!orm': 'Reform',
    'Re#orm': 'Reform',
    'Refo&m': 'Reform',
    'Reorm': 'Reform',
    
    'Revol@tion': 'Revolution',
    'Rev*lution': 'Revolution',
    'R?volut&on': 'Revolution',
    'OvrRevolution': 'Revolution',
    'WaitRevolution': 'Revolution',
    
    'Cris*s': 'Crisis',
    'Cr&sis': 'Crisis',
    'Cr@is': 'Crisis',
    
    'Agreem': 'Agreement',
    'A!reement': 'Agreement',
    'Agremet': 'Agreement',
    'grement': 'Agreement',
    
    # Special cases that might be tricky for auto-detection
    'Lang': 'Language',
    'Lss': 'Loss',
    'FinDiscovery': 'Discovery',
    'Gue': 'Guest',
    'Nat': 'Natural'
}

# Update with previously saved corrections
if 'saved_corrections' in locals() and saved_corrections:
    event_name_corrections.update(saved_corrections)
    print(f"Updated corrections dictionary with {len(saved_corrections)} saved entries")

# Automatically detect and add spelling corrections
def auto_detect_spelling_corrections(df, column='Event Name'):
    """
    Automatically detect misspelled words in the dataset and add them to the corrections dictionary.
    This uses edit distance calculation to find probable corrections.
    """
    print("Auto-detecting spelling corrections...")
    
    # Function to extract words from event names
    def extract_words(df_data, col):
        all_words = []
        for name in df_data[col].dropna():
            # Clean the name of special characters
            cleaned = re.sub(r'[^a-zA-Z\s]', '', str(name))
            words = cleaned.split()
            all_words.extend([w for w in words if len(w) > 2])  # Only include words with 3+ characters
        return all_words
    
    # Get words from the dataset
    all_words = extract_words(df, column)
    word_counts = Counter(all_words)
    
    # Get common suffixes from the dataset to use as reference (event types)
    event_suffixes = []
    for name in df[column].dropna():
        parts = str(name).split()
        if len(parts) > 1:
            event_suffixes.append(parts[-1])
    
    # Most common event types as reference for corrections
    common_event_types = [word for word, count in Counter(event_suffixes).most_common(10) if count > 1]
    
    # Build reference vocabulary from common words and event types
    # Words that appear more than 3 times are likely correct
    reference_vocab = set([word for word, count in word_counts.items() if count > 3])
    reference_vocab.update(common_event_types)
    reference_vocab.update(['Breakthrough', 'Discovery', 'Reform', 'Revolution', 'Crisis', 'Agreement'])
    
    # Find potential misspellings more efficiently
    new_corrections = {}
    rare_words = [word for word, count in word_counts.items() if 0 < count <= 5]
    
    # Pre-process the reference vocabulary for faster matching
    ref_words_by_length = {}
    for ref_word in reference_vocab:
        if len(ref_word) > 3:
            length = len(ref_word)
            if length not in ref_words_by_length:
                ref_words_by_length[length] = []
            ref_words_by_length[length].append(ref_word)
    
    # Process rare words for potential corrections
    for word in rare_words:
        word_len = len(word)
        
        # Get reference words of similar length (±2) for better performance
        candidate_lengths = [word_len-2, word_len-1, word_len, word_len+1, word_len+2]
        candidate_refs = []
        for l in candidate_lengths:
            if l in ref_words_by_length:
                candidate_refs.extend(ref_words_by_length[l])
        
        # Skip if no candidates found
        if not candidate_refs:
            continue
        
        # Find closest word in candidate references
        best_match = None
        best_similarity = 0
        
        for ref_word in candidate_refs:
            # Calculate normalized edit distance
            max_len = max(len(word), len(ref_word))
            similarity = 1 - (edit_distance(word, ref_word) / max_len)
            
            if similarity > best_similarity and similarity > 0.7 and similarity < 0.95:
                best_similarity = similarity
                best_match = ref_word
        
        # Add to corrections dictionary if found a good match
        if best_match:
            new_corrections[word] = best_match
    
    # Add new corrections to the dictionary
    added_count = 0
    for misspelled, correction in new_corrections.items():
        if misspelled not in event_name_corrections and misspelled.lower() not in event_name_corrections:
            event_name_corrections[misspelled] = correction
            added_count += 1
    
    print(f"Added {added_count} auto-detected spelling corrections")
    return added_count

# Processing order:
# 1. First load existing corrections (already done above)
# 2. Run auto-detection to find new corrections
auto_detect_spelling_corrections(df)
# 3. Now all corrections (manual + saved + auto-detected) are in event_name_corrections

# Function to clean text by removing special characters and normalizing
def clean_text(text):
    if pd.isna(text):
        return text
    
    # Replace common special characters with their normal equivalents
    char_map = {
        '$': 's', '@': 'a', '&': 'e', '!': 'i', '#': 't', '%': 'r', 
        '*': 't', '?': 'o', '\\': '', '/': '', '_': '', '-': ''
    }
    
    cleaned = str(text)
    # Replace special characters based on our mapping
    for char, replacement in char_map.items():
        cleaned = cleaned.replace(char, replacement)
        
    # Remove any remaining non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned)
    
    return cleaned

# Function to get the most similar word from a vocabulary using edit distance
def find_closest_word(word, vocabulary, threshold=0.7):
    if word in vocabulary:
        return word
    
    min_distance = float('inf')
    closest_word = word
    
    for vocab_word in vocabulary:
        # Calculate normalized edit distance
        max_len = max(len(word), len(vocab_word))
        if max_len == 0:  # Handle empty strings
            continue
        
        distance = edit_distance(word, vocab_word) / max_len
        
        # Lower distance means more similar
        if distance < min_distance and distance < (1 - threshold):
            min_distance = distance
            closest_word = vocab_word
    
    return closest_word

# Enhanced function to correct specific word patterns in event names with substring matching
def correct_specific_words(name):
    if pd.isna(name):
        return name
    
    words = name.split()
    corrected_words = []
    
    for word in words:
        # First check exact matches in our correction dictionary
        if word in event_name_corrections:
            corrected_words.append(event_name_corrections[word])
            continue
        
        # If no exact match, try to find the most similar key in our dictionary
        # This handles cases where the word is misspelled but similar to a known correction
        best_match = None
        highest_similarity = 0
        
        for key, correct_word in event_name_corrections.items():
            # Simple substring matching
            similarity = 0
            if len(key) > 2 and key in word:  # Only consider keys with length > 2
                # Calculate similarity score based on length of match relative to word
                similarity = len(key) / len(word)
            
            if similarity > highest_similarity and similarity > 0.5:  # Threshold for match
                highest_similarity = similarity
                best_match = correct_word
        
        if best_match:
            corrected_words.append(best_match)
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)

# Enhanced function to standardize event names based on patterns
def standardize_event_name(event_name):
    if pd.isna(event_name):
        return event_name
    
    event_name = str(event_name).strip()
    
    # Define patterns for event name types based on their suffixes
    event_types = ['Breakthrough', 'Crisis', 'Reform', 'Revolution', 'Agreement', 'Discovery']
    
    # Extract the prefix and suffix
    parts = event_name.split()
    if len(parts) < 2:  # If it's just one word, return as is
        return event_name
    
    prefix = " ".join(parts[:-1])  # Everything except the last word
    suffix = parts[-1]  # Last word
    
    # Find the closest suffix match from our event types
    best_match = None
    highest_similarity = 0
    
    for event_type in event_types:
        # Simple similarity measure (could be enhanced with better algorithms)
        similarity = sum(1 for a, b in zip(suffix.lower(), event_type.lower()) if a == b)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = event_type
    
    # Only replace if we have a good match
    if highest_similarity >= len(suffix) * 0.6 or highest_similarity >= len(best_match) * 0.6:
        return f"{prefix} {best_match}"
    
    return event_name

# Function to correct event names using all available corrections
def correct_event_name(name):
    """
    Apply all text cleaning and correction techniques to event names
    """
    if pd.isna(name):
        return name
    
    # First clean special characters
    cleaned_name = clean_text(name)
    
    # Split into words
    words = cleaned_name.split()
    corrected_words = []
    
    for word in words:
        # Apply word-by-word spelling correction from our dictionary
        if word in event_name_corrections:
            corrected_words.append(event_name_corrections[word])
        elif word.lower() in event_name_corrections:
            corrected_words.append(event_name_corrections[word.lower()])
        else:
            # For words not in dictionary, check for closest match
            corrected_words.append(find_closest_word(word, event_name_corrections.values()))
    
    # Rejoin the words
    return " ".join(corrected_words)

# Simplified ML approach that leverages our improved word correction system
def ml_correct_event_names(df):
    # Apply the word-by-word correction to all event names
    df['Event Name'] = df['Event Name'].apply(correct_event_name)
    
    # Add a second pass for event names that need full name standardization
    # This captures cases where the whole name structure needs fixing
    df['Event Name'] = df['Event Name'].apply(standardize_event_name)
    
    return df

# Function to standardize country values
def standardize_country(country):
    if pd.isna(country):
        return np.nan
    
    country = str(country).strip()
    
    # Map various incorrect country formats to standard ones
    if country in ['', 'Country', 'Country?', 'Country#', 'Country_', 'Country-']:
        return np.nan
    
    # If it's a valid country code (Country A-E), return as is
    if country in ['Country A', 'Country B', 'Country C', 'Country D', 'Country E']:
        return country
    
    return np.nan

# Fix the Source Confidence column (values should be between 0 and 1)
def fix_confidence(value):
    if pd.isna(value):
        return np.nan
    
    try:
        val = float(value)
        # If confidence > 1, divide by 100 if it makes sense, otherwise cap at 1
        if val > 1:
            if val < 2:  # Small values like 1.07 should be capped at 1
                return 1.0
            else:
                return val / 100  # Larger values might be percentages
        elif val < 0:  # Negative values don't make sense for confidence
            return abs(val)  # Convert to positive
        return val
    except:
        return np.nan

# Function to validate and fix years with a strict upper limit
def fix_year(year):
    max_allowed_year = 2024  # Setting a realistic upper limit
    
    if pd.isna(year):
        return np.nan
    
    try:
        year = int(float(year))
        
        # Fix years that are in the future
        if year > max_allowed_year:
            # Cap at max allowed year
            return max_allowed_year
        
        # Make sure years aren't too far in the past
        if year < 1600:
            return 1600 + (year % 100)  # Adjust to the 17th century
            
        return year
    except:
        return np.nan

# 1. Clean the Event ID column
df['Event ID'] = df['Event ID'].ffill()  # Fill missing event IDs using the previous value

# 2. Fix Event Names - multi-step process
# First clean special characters
df['Event Name'] = df['Event Name'].apply(clean_text)

# Then correct specific known misspellings
df['Event Name'] = df['Event Name'].apply(correct_specific_words)

# Then standardize based on patterns
df['Event Name'] = df['Event Name'].apply(standardize_event_name)

# Apply ML-based correction
df = ml_correct_event_names(df)

# 3. Fix Year values
df['Year'] = df['Year'].apply(fix_year)

# 4. Fix Category values
valid_categories = ['Science', 'Culture', 'Politics', 'War', 'Technology', 'Economy']
df['Category'] = df['Category'].apply(clean_text)

# Map close matches to valid categories
category_mapping = {
    'Sciece': 'Science', 'Scie': 'Science', 'Scince': 'Science', 'Science': 'Science', 
    'War': 'War', 'Wa': 'War', 'Wae': 'War',
    'Polit': 'Politics', 'Politcs': 'Politics', 'Politics': 'Politics',
    'Cultur': 'Culture', 'Cult': 'Culture', 'Culture': 'Culture',
    'Econ': 'Economy', 'Economy': 'Economy', 'Economt': 'Economy',
    'Tech': 'Technology', 'Technology': 'Technology', 'Technol': 'Technology'
}

# Apply category mapping for close matches
df['Category'] = df['Category'].apply(lambda x: next((v for k, v in category_mapping.items() 
                                                      if pd.notna(x) and k in str(x)), x))

# For remaining categories, keep only those in our valid list
df['Category'] = df['Category'].apply(lambda x: x if pd.notna(x) and x in valid_categories else np.nan)

# 5. Standardize Country column
df['Country'] = df['Country'].apply(standardize_country)

# 6. Fix Source Confidence
df['Source Confidence'] = df['Source Confidence'].apply(fix_confidence)

# 7. Remove duplicate entries - keep the first occurrence with non-null values where possible
df = df.sort_values(by=['Event ID', 'Source Confidence'], ascending=[True, False])
df = df.drop_duplicates(subset=['Event ID'], keep='first')

# 8. Additional data quality checks
# Fill missing category values based on patterns in event names, if possible
event_keywords = {
    'Science': ['breakthrough', 'discovery', 'research', 'science', 'study', 'experiment'],
    'Politics': ['politics', 'agreement', 'policy', 'government', 'international', 'law'],
    'War': ['war', 'battle', 'military', 'fight', 'crisis', 'combat'],
    'Economy': ['economy', 'financial', 'money', 'business', 'market', 'trade'],
    'Technology': ['technology', 'device', 'software', 'machine', 'technical', 'radio', 'network'],
    'Culture': ['culture', 'art', 'music', 'social', 'community', 'public']
}

def infer_category(row):
    if pd.notna(row['Category']):
        return row['Category']
    
    event_name = str(row['Event Name']).lower() if pd.notna(row['Event Name']) else ''
    
    for category, keywords in event_keywords.items():
        if any(keyword in event_name for keyword in keywords):
            return category
    
    return np.nan

df['Category'] = df.apply(infer_category, axis=1)

# 9. Additional spell correction for Event Names using most common patterns
# Extract the second word (type) from event names
event_types = [name.split()[-1] for name in df['Event Name'].dropna() if len(name.split()) > 1]
# Get most common event types (Breakthrough, Crisis, etc.)
common_types = Counter(event_types).most_common()

# Function to correct event names based on the most common patterns
def correct_event_name(name):
    if pd.isna(name) or len(name.split()) <= 1:
        return name
    
    prefix = " ".join(name.split()[:-1])  # First part of the name
    suffix = name.split()[-1]  # Type part like "Breakthrough"
    
    # Check if the suffix is close to one of our common types
    for common_type, _ in common_types:
        # Simple similarity check
        similarity = sum(1 for a, b in zip(suffix.lower(), common_type.lower()) if a == b)
        similarity_ratio = similarity / max(len(suffix), len(common_type))
        
        # If it's a close match, correct it
        if similarity_ratio > 0.7 and suffix != common_type:
            return f"{prefix} {common_type}"
    
    return name

# Apply the second level of corrections
df['Event Name'] = df['Event Name'].apply(correct_event_name)

# 10. Statistical imputation for missing values
# a. Fill missing years using median by category or event type
def impute_years_by_group(df):
    # First, try to impute based on Category
    medians_by_category = df.groupby('Category')['Year'].median().to_dict()
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_filled = df.copy()
    
    # Fill missing years with category median where possible
    for idx, row in df_filled.iterrows():
        if pd.isna(row['Year']) and pd.notna(row['Category']) and row['Category'] in medians_by_category:
            df_filled.at[idx, 'Year'] = medians_by_category[row['Category']]
    
    # For any remaining missing years, use event type median
    if df_filled['Year'].isna().sum() > 0:
        # Create an event type column
        df_filled['Event Type'] = df_filled['Event Name'].apply(lambda x: x.split()[-1] if pd.notna(x) and len(x.split()) > 0 else np.nan)
        
        # Get median by event type
        medians_by_type = df_filled.groupby('Event Type')['Year'].median().to_dict()
        
        # Fill remaining missing values
        for idx, row in df_filled.iterrows():
            if pd.isna(row['Year']) and pd.notna(row['Event Type']) and row['Event Type'] in medians_by_type:
                df_filled.at[idx, 'Year'] = medians_by_type[row['Event Type']]
        
        # Drop the temporary column
        df_filled.drop('Event Type', axis=1, inplace=True)
    
    # Fill any remaining missing years with the overall median
    overall_median = df_filled['Year'].median()
    df_filled['Year'] = df_filled['Year'].fillna(overall_median)
    
    # Round years to integers
    df_filled['Year'] = df_filled['Year'].apply(lambda x: int(x) if pd.notna(x) else x)
    
    return df_filled

# b. Use KNN imputation for missing categories based on patterns in events and years
def impute_categorical_values(df):
    # First try rule-based imputation for categories
    # Extract most common patterns between event types and categories
    df['Event Type'] = df['Event Name'].apply(lambda x: x.split()[-1] if pd.notna(x) and len(x.split()) > 0 else np.nan)
    
    # Calculate the most common category for each event type
    category_by_type = df.groupby('Event Type')['Category'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
    
    # Fill missing categories based on event type
    for idx, row in df.iterrows():
        if pd.isna(row['Category']) and pd.notna(row['Event Type']) and row['Event Type'] in category_by_type:
            df.at[idx, 'Category'] = category_by_type[row['Event Type']]
    
    # For countries, do a similar approach
    country_by_category = df.groupby('Category')['Country'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
    event_type_country = df.groupby('Event Type')['Country'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
    
    # Fill missing countries based on category or event type
    for idx, row in df.iterrows():
        if pd.isna(row['Country']):
            # Try by category first
            if pd.notna(row['Category']) and row['Category'] in country_by_category:
                df.at[idx, 'Country'] = country_by_category[row['Category']]
            # If still missing, try by event type
            elif pd.notna(row['Event Type']) and row['Event Type'] in event_type_country:
                df.at[idx, 'Country'] = event_type_country[row['Event Type']]
    
    # Drop the temporary column
    df.drop('Event Type', axis=1, inplace=True)
    
    return df

# Apply the statistical imputations
df = impute_years_by_group(df)
df = impute_categorical_values(df)

# 11. For any remaining missing values in Category and Country, use a weighted random approach
def fill_random_by_distribution(df, column):
    # Only proceed if there are missing values
    if df[column].isna().sum() > 0:
        # Get the value distribution (excluding NaN)
        value_counts = df[column].value_counts(normalize=True).to_dict()
        
        # For each missing value, select a random value based on the distribution
        for idx in df[df[column].isna()].index:
            # Generate random value based on distribution
            rand_val = np.random.choice(list(value_counts.keys()), p=list(value_counts.values()))
            df.at[idx, column] = rand_val
    
    return df

# Fill any remaining missing values in Category and Country
df = fill_random_by_distribution(df, 'Category')
df = fill_random_by_distribution(df, 'Country')

# 12. Final processing: arrange columns in the right order
df = df[['Event ID', 'Event Name', 'Year', 'Category', 'Country', 'Source Confidence']]

# 13. Save the cleaned dataset
df.to_csv('cleaned_fictional_dataset.csv', index=False)

# After saving the CSV, add this code to reprocess records with years between 2020-2024
# First, read the original corrupt dataset
corrupt_df = pd.read_csv('corrupt_fictional_dataset.csv')

# Read our newly cleaned dataset
cleaned_df = pd.read_csv('cleaned_fictional_dataset.csv')

# Find Event IDs with years between 2020-2024 in the original dataset
valid_future_years = {}
for idx, row in corrupt_df.iterrows():
    try:
        year = float(row['Year'])
        event_id = row['Event ID']
        # Check if year is between 2020-2024
        if 2020 < year <= 2024:
            valid_future_years[event_id] = int(year)
    except:
        continue

# Update the corresponding records in the cleaned dataset
updated_count = 0
for idx, row in cleaned_df.iterrows():
    event_id = row['Event ID']
    if event_id in valid_future_years:
        cleaned_df.at[idx, 'Year'] = valid_future_years[event_id]
        updated_count += 1

print(f"Updated {updated_count} records with years between 2020-2024")

# Save the updated dataset
cleaned_df.to_csv('cleaned_fictional_dataset.csv', index=False)

# Save the spelling corrections for future runs
def save_corrections():
    """Save the current spelling corrections dictionary to a JSON file"""
    try:
        # Make a copy of the dictionary to avoid modifying the original
        corrections_to_save = dict(event_name_corrections)
        
        # Filter out any non-string keys or values that might cause serialization issues
        valid_corrections = {str(k): str(v) for k, v in corrections_to_save.items() 
                           if k is not None and v is not None}
        
        # Count how many were filtered out
        filtered_count = len(corrections_to_save) - len(valid_corrections)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} invalid entries before saving")
            
        # Save to file
        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(valid_corrections, f, indent=4)
            
        print(f"Saved {len(valid_corrections)} spelling corrections to {CORRECTIONS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving spelling corrections: {e}")
        return False

# Write the cleaning results to a file
with open('cleaning_analysis.txt', 'w') as f:
    f.write(f'Original dataset shape: {pd.read_csv("corrupt_fictional_dataset.csv").shape}\n')
    f.write(f'Cleaned dataset shape: {cleaned_df.shape}\n')
    f.write(f'Number of unique Event IDs: {cleaned_df["Event ID"].nunique()}\n')
    f.write(f'Missing values after cleaning:\n')
    f.write(f'{cleaned_df.isna().sum()}\n\n')
    f.write('Sample of cleaned data:\n')
    f.write(f'{cleaned_df.head(10).to_string()}\n\n')
    
    f.write('===== Data Distribution =====\n')
    f.write('\nCategory distribution:\n')
    f.write(f'{cleaned_df["Category"].value_counts()}\n')
    f.write('\nTop countries by number of events:\n')
    f.write(f'{cleaned_df["Country"].value_counts().head()}\n')
    f.write('\nYear distribution by century:\n')
    centuries = cleaned_df['Year'].apply(lambda x: f'{int(x)//100 * 100}s' if pd.notna(x) else 'Unknown')
    f.write(f'{centuries.value_counts()}\n')
    f.write(f'\nAverage source confidence: {cleaned_df["Source Confidence"].mean():.2f}\n')
    
    # Additional analysis
    f.write('\n===== Data Quality Analysis =====\n')
    f.write(f'\nPercentage of missing values by column:\n')
    for col in cleaned_df.columns:
        pct_missing = cleaned_df[col].isna().mean() * 100
        f.write(f'{col}: {pct_missing:.2f}%\n')
    
    f.write('\nYear range (excluding missing values):\n')
    f.write(f'Min year: {cleaned_df["Year"].min()}, Max year: {cleaned_df["Year"].max()}\n')
    
    # New analysis of corrections
    f.write(f'\nEvent types after correction:\n')
    corrected_event_types = [name.split()[-1] for name in cleaned_df['Event Name'].dropna() if len(name.split()) > 1]
    f.write(f'{Counter(corrected_event_types).most_common()}\n')
    
    # New analysis specifically about years
    f.write(f'\nNumber of events by year range:\n')
    year_ranges = {
        '1600-1699': ((cleaned_df['Year'] >= 1600) & (cleaned_df['Year'] <= 1699)).sum(),
        '1700-1799': ((cleaned_df['Year'] >= 1700) & (cleaned_df['Year'] <= 1799)).sum(),
        '1800-1899': ((cleaned_df['Year'] >= 1800) & (cleaned_df['Year'] <= 1899)).sum(),
        '1900-1999': ((cleaned_df['Year'] >= 1900) & (cleaned_df['Year'] <= 1999)).sum(),
        '2000-2020': ((cleaned_df['Year'] >= 2000) & (cleaned_df['Year'] <= 2020)).sum(),
        '2021-2024': ((cleaned_df['Year'] >= 2021) & (cleaned_df['Year'] <= 2024)).sum()
    }
    for range_name, count in year_ranges.items():
        f.write(f'{range_name}: {count}\n')

# Save corrections for future runs
save_corrections()

print(f"Cleaning completed. Results saved to 'cleaned_fictional_dataset.csv'")
print(f"Cleaning analysis saved to 'cleaning_analysis.txt'") 