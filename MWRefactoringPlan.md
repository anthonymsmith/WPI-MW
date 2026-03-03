# Music Worcester Analytics - Refactoring Guide
*Created: 2024-01-12*
*Claude conversation preservation*

## Table of Contents
1. [Overview](#overview)
2. [Phase 0: Preparation](#phase-0-preparation)
3. [Phase 1: Security Fixes](#phase-1-security-fixes)
4. [Phase 2: Code Hygiene](#phase-2-code-hygiene)
5. [Phase 3: Input Validation](#phase-3-input-validation)
6. [Phase 4: Configuration](#phase-4-configuration)
7. [Phase 5: Documentation](#phase-5-documentation)
8. [Phase 6: Function Decomposition](#phase-6-function-decomposition)
9. [Testing Strategy](#testing-strategy)
10. [Working with Claude (Memory Guide)](#working-with-claude)
11. [Rollback Procedures](#rollback-procedures)
12. [Key Scripts](#key-scripts)

---

## Overview

This is a staged, incremental refactoring plan for the Music Worcester patron analytics system. The goal is to improve code quality, security, and maintainability without breaking existing functionality.

**Key Principle**: Validate after every change. Never proceed to the next phase until current phase passes validation.

### Current System
- **MWSalesSumm.ipynb**: Main orchestration notebook
- **MW_functions.py**: Data loading/cleaning (older, needs cleanup)
- **Models_functions.py**: Patron modeling and scoring (newer, cleaner)

### Critical Issues Identified
1. 🔴 Hardcoded Google API key (security risk)
2. 🔴 Import mismatch: `Model_functions` vs `Models_functions`
3. 🟡 Walk-up sales all assigned same AccountId
4. 🟡 Duplicate `calculate_event_scores` function
5. 🟡 200+ lines of commented debug code
6. 🟡 No input validation
7. 🟡 Magic numbers throughout code

---

## Phase 0: Preparation

**Goal**: Create safety net before any changes
**Risk**: None
**Time**: 2 hours
**Can Skip**: ❌ Never

### Steps

#### 1. Create Git Repository
```bash
cd /Users/antho/Documents/WPI-MW
git init
git add .
git commit -m "Initial commit - baseline before refactoring"
git branch refactoring-backup
git checkout -b phase1-security-fixes
```

#### 2. Create Baseline Outputs
```bash
# Run current pipeline
python MWSalesSumm.ipynb

# Save outputs as baseline
cp DataMerge.csv DataMerge_baseline.csv
cp Patrons.csv Patrons_baseline.csv
cp RetChurnRates.csv RetChurnRates_baseline.csv
```

#### 3. Create Validation Script

Save as `validate_outputs.py`:
```python
"""
Output validation script for Music Worcester analytics pipeline.
Compares current outputs against baseline to ensure no regression.
"""
import pandas as pd
import numpy as np
import sys

def compare_dataframes(df1, df2, name, tolerance=1e-6):
    """Compare two DataFrames and report differences."""
    print(f"\n=== Comparing {name} ===")
    
    # Shape check
    if df1.shape != df2.shape:
        print(f"❌ SHAPE MISMATCH: {df1.shape} vs {df2.shape}")
        return False
    else:
        print(f"✅ Shape matches: {df1.shape}")
    
    # Column check
    if set(df1.columns) != set(df2.columns):
        print(f"❌ COLUMN MISMATCH")
        print(f"   Missing: {set(df2.columns) - set(df1.columns)}")
        print(f"   Extra: {set(df1.columns) - set(df2.columns)}")
        return False
    else:
        print(f"✅ Columns match: {len(df1.columns)} columns")
    
    # Value check (for numeric columns)
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    max_diff = 0
    problematic_cols = []
    
    for col in numeric_cols:
        diff = np.abs(df1[col] - df2[col]).max()
        if diff > tolerance:
            problematic_cols.append((col, diff))
            max_diff = max(max_diff, diff)
    
    if problematic_cols:
        print(f"❌ NUMERIC DIFFERENCES FOUND:")
        for col, diff in problematic_cols[:5]:  # Show first 5
            print(f"   {col}: max diff = {diff}")
        return False
    else:
        print(f"✅ All numeric values match (tolerance={tolerance})")
    
    print(f"✅ {name} validation PASSED")
    return True

def validate_pipeline():
    """Validate pipeline outputs against baseline."""
    try:
        print("="*60)
        print("Music Worcester Analytics - Output Validation")
        print("="*60)
        
        # Load baseline
        print("\nLoading baseline outputs...")
        baseline_merge = pd.read_csv('DataMerge_baseline.csv', low_memory=False)
        baseline_patrons = pd.read_csv('Patrons_baseline.csv', low_memory=False)
        baseline_rates = pd.read_csv('RetChurnRates_baseline.csv')
        print("✅ Baseline files loaded")
        
        # Load current
        print("\nLoading current outputs...")
        current_merge = pd.read_csv('DataMerge.csv', low_memory=False)
        current_patrons = pd.read_csv('Patrons.csv', low_memory=False)
        current_rates = pd.read_csv('RetChurnRates.csv')
        print("✅ Current files loaded")
        
        # Compare
        all_passed = True
        all_passed &= compare_dataframes(baseline_merge, current_merge, "DataMerge")
        all_passed &= compare_dataframes(baseline_patrons, current_patrons, "Patrons")
        all_passed &= compare_dataframes(baseline_rates, current_rates, "RetChurnRates")
        
        # Final result
        print("\n" + "="*60)
        if all_passed:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("="*60)
            return 0
        else:
            print("❌ SOME VALIDATIONS FAILED")
            print("="*60)
            return 1
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you've run the pipeline and created baseline files.")
        return 1
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = validate_pipeline()
    sys.exit(exit_code)
```

#### 4. Test Validation
```bash
python validate_outputs.py
# Should show: 🎉 ALL VALIDATIONS PASSED!
```

#### 5. Create Project Documentation

Save as `docs/project_context.md`:
```markdown
# Project Context - Music Worcester Analytics

## Quick Reference
**Project**: Patron analytics for Music Worcester arts organization
**Language**: Python 3.x
**Main Files**: MWSalesSumm.ipynb, MW_functions.py, Models_functions.py
**Status**: Production system, needs refactoring

## System Overview
Processes Salesforce ticketing data to generate:
- RFM scores (Recency, Frequency, Monetary)
- Patron segments (New, High, Slipping, etc.)
- Genre/venue/class preferences
- Retention and churn rates
- Geographic distribution

## Key Concepts
- **Fiscal Year**: July 1 - June 30
- **Burst Collapse**: 4-day window for festival attendance
- **Walk-up Sales**: Tickets without account information
- **Subscription**: Season ticket package

## Data Sources
1. **SalesforceLatest.csv**: Raw transaction data
2. **EventManifest.xlsx**: Event metadata (manual)
3. **EventPnL.xlsx**: Financial data (manual)
4. **Worcester Chorus current members.xlsx**: Chorus roster
5. **BoardCorporators.csv**: Board/corporator list
6. **final_regions.csv**: Geographic region mapping

## Output Files
1. **DataMerge.csv**: Transaction-level processed data
2. **Patrons.csv**: Patron-level analytics
3. **RetChurnRates.csv**: Retention metrics by fiscal year
4. **anon_*.csv**: Anonymized versions (PII removed)

## Known Issues
See refactoring_plan.md for full list

## Current Refactoring Status
- Phase 0: ✅ COMPLETE
- Phase 1: 🔄 IN PROGRESS
```

---

## Phase 1: Security Fixes

**Goal**: Fix security vulnerabilities without changing logic
**Risk**: Low
**Time**: 1 week
**Can Skip**: ❌ Never

### Critical Security Issues
1. 🔴 Hardcoded Google API key exposed in code
2. 🔴 Import name mismatch causing potential failures

### Changes Required

#### 1. Create Configuration File

Save as `config.py`:
```python
"""
Configuration management for Music Worcester analytics.

Environment Variables Required:
    GOOGLE_MAPS_API_KEY: Google Maps Geocoding API key
    USPS_USER_ID: USPS ZIP+4 lookup API user ID (optional)
"""
import os
from pathlib import Path

class Config:
    """Main configuration class."""
    
    # API Keys (from environment)
    GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
    USPS_USER_ID = os.environ.get('USPS_USER_ID')
    
    @classmethod
    def validate(cls):
        """Validate that required configuration exists."""
        missing = []
        
        if not cls.GOOGLE_MAPS_API_KEY:
            missing.append('GOOGLE_MAPS_API_KEY')
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}\n"
                f"Set them in your environment or .env file.\n"
                f"Example: export GOOGLE_MAPS_API_KEY='your-key-here'"
            )
        return True
    
    @classmethod
    def summary(cls):
        """Print configuration summary (without exposing keys)."""
        print("Configuration:")
        print(f"  GOOGLE_MAPS_API_KEY: {'✅ Set' if cls.GOOGLE_MAPS_API_KEY else '❌ Missing'}")
        print(f"  USPS_USER_ID: {'✅ Set' if cls.USPS_USER_ID else '⚠️  Optional'}")

# Validate on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")
```

#### 2. Create .env Template

Save as `.env.example`:
```bash
# Music Worcester Analytics - Environment Variables
# Copy this file to .env and fill in your values
# NEVER commit .env to git!

# Google Maps Geocoding API Key (REQUIRED)
# Get from: https://console.cloud.google.com/apis/credentials
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# USPS User ID (OPTIONAL - for ZIP+4 lookup)
USPS_USER_ID=your_usps_user_id_here
```

#### 3. Update .gitignore

Add to `.gitignore`:
```
# Environment variables (NEVER COMMIT!)
.env

# Baseline files (large, regenerate as needed)
*_baseline.csv

# API keys or credentials
*_api_key.txt
credentials.json
```

#### 4. Update MW_functions.py

Find this function:
```python
# OLD CODE - DELETE THIS
def get_geocode_info(address):
    google_api_key = 'AIzaSyAC4jkZD-p7bkor1InDTyw2Q2ULXK23yLw'  # ❌ REMOVE!
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    # ...
```

Replace with:
```python
# NEW CODE
from config import Config

def get_geocode_info(address):
    """
    Get geocoding information from Google Maps API.
    
    Args:
        address (str): Full address to geocode
        
    Returns:
        dict: {'lat': latitude, 'lng': longitude} or None if failed
        
    Raises:
        ValueError: If API key not configured
    """
    google_api_key = Config.GOOGLE_MAPS_API_KEY
    if not google_api_key:
        raise ValueError("Google Maps API key not configured")
    
    base_url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': address, 'key': google_api_key}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        res_json = response.json()
        
        if 'results' in res_json and len(res_json['results']) > 0:
            return res_json['results'][0]['geometry']['location']
        else:
            return None
            
    except RequestException as e:
        print(f"Error during geocoding request: {str(e)}")
        return None
    except ValueError as e:
        print(f"Error parsing geocoding response: {str(e)}")
        return None
    except KeyError as e:
        print(f"Unexpected response format: {str(e)}")
        return None
```

#### 5. Fix Import Name in MW_functions.py

Find:
```python
# OLD
import Model_functions as mod  # ❌ WRONG NAME!
```

Replace with:
```python
# NEW
import Models_functions as mod  # ✅ CORRECT NAME
```

#### 6. Update MWSalesSumm.ipynb

At the top of the notebook, add:
```python
from config import Config

# Validate configuration before starting
try:
    Config.validate()
    Config.summary()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    sys.exit(1)
```

#### 7. Set Environment Variables
```bash
# Option A: Export in terminal (temporary)
export GOOGLE_MAPS_API_KEY='your-new-key-here'

# Option B: Create .env file (recommended)
cp .env.example .env
# Edit .env and add your keys

# Option C: Add to ~/.bashrc or ~/.zshrc (persistent)
echo 'export GOOGLE_MAPS_API_KEY="your-new-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Testing Phase 1
```bash
# 1. Test configuration
python -c "from config import Config; Config.validate(); print('✅ Config OK')"

# 2. Run pipeline
python MWSalesSumm.ipynb

# 3. Validate outputs
python validate_outputs.py

# 4. If all pass, commit
git add .
git commit -m "Phase 1: Security fixes

- Externalized Google Maps API key to environment variable
- Created config.py with Config class
- Added .env.example template
- Fixed import: Model_functions -> Models_functions
- Updated .gitignore to protect credentials

Validation: ✅ All outputs match baseline
Next: Phase 2 - Code hygiene"
```

### ⚠️ IMPORTANT: Revoke Old API Key

**IMMEDIATELY after deploying Phase 1:**

1. Go to Google Cloud Console: https://console.cloud.google.com/apis/credentials
2. Find key: `AIzaSyAC4jkZD-p7bkor1InDTyw2Q2ULXK23yLw`
3. Click "Delete" or "Regenerate"
4. Update your .env file with new key

---

## Phase 2: Code Hygiene

**Goal**: Clean up without changing logic
**Risk**: Low
**Time**: 1 week
**Can Skip**: ⚠️ If pressed for time

### Changes Required

#### 1. Remove Commented Debug Code

Save as `scripts/cleanup_script.py`:
```python
"""
Remove commented debug code from Python files.
Only removes lines matching specific patterns.
"""
import re
import sys
from pathlib import Path

def clean_file(filepath, dry_run=True):
    """
    Remove commented debug lines from a Python file.
    
    Removes lines that match:
    - #logger.debug(...)
    - #logger.info(...)  (if commented)
    - # logger.debug(...)
    
    Args:
        filepath: Path to Python file
        dry_run: If True, only show what would be removed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    removed_lines = []
    
    patterns = [
        r'^\s*#\s*logger\.(debug|info)',  # Commented logger calls
        r'^\s*#\s*print\(',                # Commented prints (optional)
    ]
    
    for i, line in enumerate(lines, 1):
        should_remove = any(re.match(pattern, line) for pattern in patterns)
        
        if should_remove:
            removed_lines.append((i, line.rstrip()))
        else:
            cleaned_lines.append(line)
    
    # Report
    print(f"\nLines to remove: {len(removed_lines)}")
    if removed_lines:
        print("\nPreview (first 10):")
        for line_num, content in removed_lines[:10]:
            print(f"  Line {line_num}: {content[:70]}...")
    
    if dry_run:
        print(f"\n⚠️  DRY RUN - No changes made")
        print(f"Run with --execute to apply changes")
        return True
    
    # Write changes
    with open(filepath, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"\n✅ Removed {len(removed_lines)} lines from {filepath}")
    return True

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean commented debug code')
    parser.add_argument('files', nargs='+', help='Python files to clean')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually modify files (default is dry run)')
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified")
    else:
        response = input("\n⚠️  This will modify files. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return 1
    
    for filepath in args.files:
        clean_file(filepath, dry_run=dry_run)
    
    if dry_run:
        print("\n" + "="*60)
        print("To apply changes, run with --execute flag")
        print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

#### 2. Run Cleanup (Dry Run First)
```bash
# Dry run to see what would be removed
python scripts/cleanup_script.py MW_functions.py Models_functions.py

# Review the output carefully!

# If looks good, execute
python scripts/cleanup_script.py MW_functions.py Models_functions.py --execute
```

#### 3. Add Logging for Filtered Data

In `MW_functions.py`, update functions to log excluded data:
```python
def event_counts(df, logger, event_column):
    """Count events by category for each account."""
    start = perf_counter()
    df = df.copy()  # ✅ Explicit copy
    
    # Fill NaN values
    df[event_column] = df[event_column].fillna('None')
    
    # Filter relevant event types
    initial_count = len(df)
    df_filtered = df[df['EventType'].isin(['Live', 'Virtual', 'Subscriptions'])]
    excluded_count = initial_count - len(df_filtered)
    
    # ✅ NEW: Log excluded data
    if excluded_count > 0:
        excluded_types = df[~df['EventType'].isin(['Live', 'Virtual', 'Subscriptions'])]['EventType'].value_counts()
        logger.info(f"Excluded {excluded_count} records. Event types: {dict(excluded_types)}")
    
    # ... rest of function unchanged
```

Apply similar pattern to:
- `sales_initial_prep()` - log deleted tickets removed
- `venue_and_attribute_processing()` - log venue consolidations
- `state_and_city_processing()` - log corrections made

#### 4. Fix DataFrame Copy Warnings

Pattern to apply throughout:
```python
# OLD - inconsistent
def some_function(df, logger):
    df['Column'] = df['Column'].fillna('None')  # May cause warning
    df.loc[mask, 'Column'] = 'Value'  # May cause warning

# NEW - explicit copy
def some_function(df, logger):
    df = df.copy()  # ✅ Explicit copy at start
    df['Column'] = df['Column'].fillna('None')  # Safe
    df.loc[mask, 'Column'] = 'Value'  # Safe
    return df  # Return modified copy
```

### Testing Phase 2
```bash
# Run pipeline
python MWSalesSumm.ipynb

# Validate
python validate_outputs.py

# Check git diff to review changes
git diff MW_functions.py

# Commit if passed
git add .
git commit -m "Phase 2: Code hygiene

- Removed 200+ lines of commented debug code
- Added logging for excluded/filtered data
- Fixed DataFrame copy warnings with explicit .copy()
- Improved data transparency in logs

Validation: ✅ All outputs match baseline
Next: Phase 3 - Input validation"
```

---

## Phase 3: Input Validation

**Goal**: Add safety checks without changing logic
**Risk**: Low
**Time**: 1 week
**Can Skip**: ⚠️ Recommended to keep

### Changes Required

#### Create validation.py
```python
"""
Input validation utilities for Music Worcester analytics.

Provides consistent error handling and data quality checks
throughout the pipeline.
"""
import pandas as pd
import os
from pathlib import Path

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_file_exists(filepath, name="File"):
    """
    Validate that a file exists and is readable.
    
    Args:
        filepath: Path to file
        name: Descriptive name for error messages
        
    Raises:
        ValidationError: If file doesn't exist or isn't readable
    """
    if not os.path.exists(filepath):
        raise ValidationError(f"{name} not found: {filepath}")
    
    if not os.access(filepath, os.R_OK):
        raise ValidationError(f"{name} exists but is not readable: {filepath}")
    
    return True

def validate_dataframe_not_empty(df, name="DataFrame"):
    """
    Validate that a DataFrame has data.
    
    Args:
        df: pandas DataFrame
        name: Descriptive name for error messages
        
    Raises:
        ValidationError: If DataFrame is empty
    """
    if df is None:
        raise ValidationError(f"{name} is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} is not a DataFrame (type: {type(df)})")
    
    if df.empty:
        raise ValidationError(f"{name} is empty")
    
    if len(df) == 0:
        raise ValidationError(f"{name} has no rows")
    
    return True

def validate_required_columns(df, required_cols, name="DataFrame"):
    """
    Validate that required columns exist in DataFrame.
    
    Args:
        df: pandas DataFrame
        required_cols: List of required column names
        name: Descriptive name for error messages
        
    Raises:
        ValidationError: If required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{name} missing required columns: {sorted(missing)}\n"
            f"Available columns: {sorted(df.columns)}"
        )
    return True

def validate_numeric_range(df, column, min_val=None, max_val=None, name=None):
    """
    Validate that numeric column values are in expected range.
    
    Args:
        df: pandas DataFrame
        column: Column name to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Descriptive name for error messages
        
    Returns:
        dict: Statistics about violations found
    """
    name = name or column
    
    if column not in df.columns:
        raise ValidationError(f"Column '{column}' not found in DataFrame")
    
    issues = {}
    
    # Check for non-numeric
    non_numeric = pd.to_numeric(df[column], errors='coerce').isna().sum()
    if non_numeric > 0:
        issues['non_numeric'] = non_numeric
    
    numeric_col = pd.to_numeric(df[column], errors='coerce')
    
    # Check range
    if min_val is not None:
        below_min = (numeric_col < min_val).sum()
        if below_min > 0:
            issues['below_minimum'] = below_min
    
    if max_val is not None:
        above_max = (numeric_col > max_val).sum()
        if above_max > 0:
            issues['above_maximum'] = above_max
    
    return issues

def validate_sales_data(df, logger):
    """
    Comprehensive validation for sales data.
    
    Args:
        df: Sales DataFrame
        logger: Logger instance
        
    Returns:
        bool: True if validation passed (with warnings logged)
        
    Raises:
        ValidationError: If critical validation fails
    """
    logger.info("Validating sales data...")
    
    # Required columns
    required = [
        'OrderNumber', 'AccountName', 'EventName_sales', 
        'EventDate_sales', 'Quantity', 'ItemPrice'
    ]
    validate_required_columns(df, required, "Sales data")
    
    # Check for completely null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        logger.warning(f"Columns with all null values: {null_cols}")
    
    # Numeric validations
    qty_issues = validate_numeric_range(df, 'Quantity', min_val=0)
    if qty_issues:
        logger.warning(f"Quantity validation issues: {qty_issues}")
    
    price_issues = validate_numeric_range(df, 'ItemPrice', min_val=0)
    if price_issues:
        logger.warning(f"ItemPrice validation issues: {price_issues}")
    
    # Data quality summary
    logger.info(f"✅ Sales data validation passed: {len(df):,} records")
    logger.info(f"   Unique accounts: {df['AccountName'].nunique():,}")
    logger.info(f"   Date range: {df['EventDate_sales'].min()} to {df['EventDate_sales'].max()}")
    
    return True

def validate_event_data(df, logger):
    """
    Validate event manifest data.
    
    Args:
        df: Event DataFrame
        logger: Logger instance
        
    Returns:
        bool: True if validation passed
    """
    logger.info("Validating event data...")
    
    required = ['EventId', 'EventName', 'EventDate', 'EventVenue']
    validate_required_columns(df, required, "Event data")
    
    # Check for duplicates
    duplicates = df['EventId'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate EventIds")
    
    logger.info(f"✅ Event data validation passed: {len(df):,} events")
    return True

# Convenience wrappers for common operations
def validate_and_load_csv(filepath, name, logger, **kwargs):
    """
    Validate file exists, load CSV, and validate result.
    
    Args:
        filepath: Path to CSV file
        name: Descriptive name
        logger: Logger instance
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded and validated DataFrame
    """
    validate_file_exists(filepath, name)
    
    try:
        df = pd.read_csv(filepath, **kwargs)
    except Exception as e:
        raise ValidationError(f"Failed to load {name}: {e}")
    
    validate_dataframe_not_empty(df, name)
    
    logger.info(f"✅ Loaded {name}: {df.shape}")
    return df

def validate_and_load_excel(filepath, name, logger, **kwargs):
    """
    Validate file exists, load Excel, and validate result.
    
    Args:
        filepath: Path to Excel file
        name: Descriptive name
        logger: Logger instance
        **kwargs: Additional arguments for pd.read_excel
        
    Returns:
        pd.DataFrame: Loaded and validated DataFrame
    """
    validate_file_exists(filepath, name)
    
    try:
        df = pd.read_excel(filepath, **kwargs)
    except Exception as e:
        raise ValidationError(f"Failed to load {name}: {e}")
    
 #### Update MW_functions.py to Use Validation
````python
# At top of MW_functions.py
from validation import (
    ValidationError,
    validate_file_exists,
    validate_dataframe_not_empty,
    validate_required_columns,
    validate_sales_data,
    validate_event_data,
    validate_and_load_csv,
    validate_and_load_excel
)

# Update load functions:
def load_event_manifest(manifest_file, logger):
    """Load and validate event manifest."""
    start = perf_counter()
    
    # Use validated loader
    event_df = validate_and_load_excel(
        manifest_file, 
        "Event manifest",
        logger,
        sheet_name='EventManifest'
    )
    
    # ... rest of function unchanged
    
    # Validate result
    validate_event_data(event_df, logger)
    
    # ... timing and return unchanged
````

#### Update MWSalesSumm.ipynb
````python
from validation import ValidationError

def main(...):
    try:
        # Validate files exist before starting
        logger.info("Pre-flight checks...")
        validate_file_exists(manifest_file, "Event manifest")
        validate_file_exists(PnL_file, "P&L file")
        validate_file_exists(sales_file, "Sales file")
        validate_file_exists(regions_file, "Regions file")
        logger.info("✅ All input files exist")
        
        # ... rest of pipeline
        
    except ValidationError as e:
        logger.error(f"❌ Validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
````

### Testing Phase 3
````bash
# Test with good data
python MWSalesSumm.ipynb
python validate_outputs.py

# Test with bad data (create test scenarios)
# 1. Rename a file temporarily
mv SalesforceLatest.csv SalesforceLatest.csv.backup
python MWSalesSumm.ipynb  # Should fail gracefully with clear error
mv SalesforceLatest.csv.backup SalesforceLatest.csv

# Commit if passed
git add .
git commit -m "Phase 3: Added input validation

- Created validation.py with comprehensive checks
- Added file existence validation
- Added DataFrame structure validation  
- Added numeric range checks
- Wrapped all file loads with validation
- Improved error messages

Validation: ✅ All outputs match baseline
Next: Phase 4 - Configuration externalization"
````

---

## Phase 4: Configuration Externalization

**Goal**: Move magic numbers to config
**Risk**: Low
**Time**: 1 week
**Can Skip**: ✅ If time constrained

### Expand config.py
````python
# config.py (add to existing file)

class PatronAnalyticsConfig:
    """Configuration for patron analytics calculations."""
    
    # ==================== Bulk Buyer Detection ====================
    BULK_BUYER_TICKET_THRESHOLD = 12
    """Minimum tickets per event to qualify as bulk buyer.
    
    Reasoning: Based on historical analysis, 12 tickets typically
    represents group organizers (schools, senior centers, community
    organizations) while excluding typical family purchases (4-6 tickets).
    
    Adjust if seeing too many/few patrons flagged as bulk buyers.
    """
    
    FREQUENT_BULK_BUYER_EVENT_THRESHOLD = 4
    """Minimum events with bulk purchases to qualify as frequent bulk buyer.
    
    Reasoning: Patrons who organize 4+ group outings per year represent
    valuable institutional relationships worthy of targeted engagement.
    
    Consider lowering to 3 if missing important group organizers.
    """
    
    # ==================== Patron Lifecycle ====================
    NEW_PATRON_DAYS_THRESHOLD = 250
    """Days since first event to still be considered 'new'.
    
    Reasoning: ~8 months (250 days) gives new patrons time to experience
    multiple seasons and programming variety before being moved to other
    lifecycle segments.
    
    Typical range: 180-365 days depending on season length.
    """
    
    REENGAGED_YEARS_THRESHOLD = 2.5
    """Years since previous event to qualify as 're-engaged'.
    
    Reasoning: A gap of 2.5+ years indicates significant lapse in attendance
    followed by return to engagement. These patrons need special attention.
    
    Consider 2.0 if want to flag re-engagement earlier.
    """
    
    # ==================== RFM Scoring Bins ====================
    # Recency (days since last event)
    RECENCY_BINS = [-1, 120, 400, 700, 1500, 2000, float('inf')]
    RECENCY_LABELS = [5, 4, 3, 2, 1, 0]
    """
    Recency scoring:
    - 5: 0-120 days (4 months) - Very recent
    - 4: 120-400 days (13 months) - Recent
    - 3: 400-700 days (23 months) - Moderate
    - 2: 700-1500 days (4 years) - Distant
    - 1: 1500-2000 days (5.5 years) - Very distant
    - 0: 2000+ days - Dormant
    """
    
    # Frequency (number of distinct events attended)
    FREQUENCY_BINS = [-1, 1, 3, 5, 8, 11, float('inf')]
    FREQUENCY_LABELS = [0, 1, 2, 3, 4, 5]
    """
    Frequency scoring:
    - 0: 0-1 events - Single visit
    - 1: 2-3 events - Occasional
    - 2: 4-5 events - Regular
    - 3: 6-8 events - Frequent
    - 4: 9-11 events - Very frequent
    - 5: 12+ events - Super fan
    """
    
    # Monetary (total ticket spend)
    MONETARY_BINS = [-1, 10, 80, 200, 400, 1000, float('inf')]
    MONETARY_LABELS = [0, 1, 2, 3, 4, 5]
    """
    Monetary scoring:
    - 0: $0-10 - Comp/minimal
    - 1: $10-80 - Low spend
    - 2: $80-200 - Moderate
    - 3: $200-400 - Good
    - 4: $400-1000 - High value
    - 5: $1000+ - VIP
    """
    
    # ==================== Event Analysis ====================
    BURST_COLLAPSE_DAYS = 4
    """Days within which events are considered a 'burst' (festival weekend).
    
    Reasoning: A 4-day window captures typical weekend festival patterns
    (Thu-Sun) without over-collapsing distinct attendance patterns.
    
    Used to prevent festival weekends from artificially inflating frequency
    scores. Only the first event in a burst is counted.
    """
    
    VENUE_MINIMUM_OCCURRENCES = 6
    """Minimum events at a venue to include in preference analysis.
    
    Reasoning: Filters out one-off venues that don't represent meaningful
    venue preferences. Venues with <6 events are usually special occasions.
    
    Consider lowering to 4-5 if losing important venue insights.
    """
    
    ENTROPY_THRESHOLD = 1.1
    """Threshold for classifying patron as 'Omnivore' based on genre entropy.
    
    Reasoning: Entropy >1.1 indicates very diverse attendance patterns
    across genres, suggesting no strong preference.
    
    Lower values (0.9-1.0) = stricter omnivore definition
    Higher values (1.2-1.5) = looser omnivore definition
    """
    
    # ==================== Data Quality ====================
    YEARS_OF_DATA_TO_KEEP = 15
    """Number of years of historical data to retain in analysis.
    
    Reasoning: 15 years provides sufficient history for trend analysis
    while keeping data volume manageable.
    
    Minimum recommended: 5 years for retention analysis
    """
    
    RFM_SCORE_THRESHOLD_FOR_GEOCODING = 0
    """Minimum RFM score required to geocode address.
    
    Reasoning: Set to 0 to geocode all patrons. Increase to 3-5 to
    skip geocoding for very low-value patrons (saves API calls/cost).
    
    Current setting geocodes all patrons regardless of RFM score.
    """
    
    # ==================== CLV Calculation ====================
    CLV_WEIGHT_RECENCY = 0.2
    CLV_WEIGHT_FREQUENCY = 0.3
    CLV_WEIGHT_MONETARY = 0.5
    """Weights for Customer Lifetime Value calculation.
    
    Reasoning: Monetary value weighted highest as it directly impacts
    revenue. Frequency second as it indicates engagement. Recency lowest
    as it's already captured in segment classification.
    
    Weights must sum to 1.0
    """
    
    # ==================== Growth Score ====================
    GROWTH_SCORE_WEIGHT_MAP = {
        0: 0.40,   # Current year
        -1: 0.30,  # Last year
        -2: 0.20,  # 2 years ago
        -3: 0.10,  # 3 years ago
        -4: 0.05   # 4 years ago
    }
    """Weights for calculating monetary growth trend.
    
    Reasoning: Recent years weighted more heavily to capture current
    trajectory rather than ancient history.
    """
    
    # ==================== Regularity Weights ====================
    REGULARITY_WEIGHT_SEASON_COUNT = 0.4
    REGULARITY_WEIGHT_CLUSTER_FREQUENCY = 0.2
    REGULARITY_WEIGHT_EVENTS_PER_SEASON = 0.4
    """Weights for calculating patron regularity score.
    
    Components:
    - Season count: Number of different seasons attended (consistency)
    - Cluster frequency: Number of distinct visit occasions per season
    - Events per season: Average events attended when they do attend
    
    Weights must sum to 1.0
    """
    
    # ==================== Preference Strength ====================
    PREFERENCE_CONFIDENCE_ALPHA = 0.4
    """Weight parameter for preference confidence calculation.
    
    Formula: confidence = (1-α)*preference_strength + α*event_weighting
    
    - Lower α (0.2-0.3): Emphasize genre distribution over attendance count
    - Higher α (0.5-0.6): Emphasize attendance count over distribution
    - Current 0.4: Balanced approach
    """
    
    @classmethod
    def validate_weights(cls):
        """Validate that weights sum to 1.0 where required."""
        clv_sum = (cls.CLV_WEIGHT_RECENCY + 
                   cls.CLV_WEIGHT_FREQUENCY + 
                   cls.CLV_WEIGHT_MONETARY)
        if abs(clv_sum - 1.0) > 0.001:
            raise ValueError(f"CLV weights must sum to 1.0, got {clv_sum}")
        
        reg_sum = (cls.REGULARITY_WEIGHT_SEASON_COUNT +
                   cls.REGULARITY_WEIGHT_CLUSTER_FREQUENCY +
                   cls.REGULARITY_WEIGHT_EVENTS_PER_SEASON)
        if abs(reg_sum - 1.0) > 0.001:
            raise ValueError(f"Regularity weights must sum to 1.0, got {reg_sum}")
        
        return True
    
    @classmethod
    def to_dict(cls):
        """Export config as dictionary for logging/documentation."""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }
    
    @classmethod
    def summary(cls):
        """Print configuration summary."""
        print("\nPatron Analytics Configuration:")
        print("="*60)
        print(f"Bulk Buyer Threshold: {cls.BULK_BUYER_TICKET_THRESHOLD} tickets")
        print(f"New Patron Window: {cls.NEW_PATRON_DAYS_THRESHOLD} days")
        print(f"Re-engaged Gap: {cls.REENGAGED_YEARS_THRESHOLD} years")
        print(f"Burst Collapse: {cls.BURST_COLLAPSE_DAYS} days")
        print(f"Data History: {cls.YEARS_OF_DATA_TO_KEEP} years")
        print("="*60)

# Validate on module load
PatronAnalyticsConfig.validate_weights()
````

### Update Functions to Use Config
````python
# MW_functions.py

from config import PatronAnalyticsConfig as PAC

def add_bulk_buyers(df, logger):
    """Identify bulk and frequent bulk buyers."""
    start = perf_counter()
    
    # Use config values
    bulk_threshold = PAC.BULK_BUYER_TICKET_THRESHOLD
    event_count_threshold = PAC.FREQUENT_BULK_BUYER_EVENT_THRESHOLD
    
    logger.debug(f"Using bulk buyer thresholds: {bulk_threshold} tickets, "
                 f"{event_count_threshold} events")
    
    # ... rest of function unchanged

# Models_functions.py

from config import PatronAnalyticsConfig as PAC

def calculate_event_scores(df, logger, event_column, 
                          venue_threshold=None, burst_days=None):
    """Calculate preference scores with configurable parameters."""
    
    # Use config defaults if not specified
    if venue_threshold is None:
        venue_threshold = PAC.VENUE_MINIMUM_OCCURRENCES
    if burst_days is None:
        burst_days = PAC.BURST_COLLAPSE_DAYS
    
    logger.info(f"Calculating {event_column} scores with "
                f"venue_threshold={venue_threshold}, "
                f"burst_days={burst_days}")
    
    # ... rest of function unchanged
````

### Testing Phase 4
````bash
# Test configuration validation
python -c "from config import PatronAnalyticsConfig; PatronAnalyticsConfig.validate_weights(); print('✅ Config valid')"

# Run pipeline
python MWSalesSumm.ipynb

# Validate outputs
python validate_outputs.py

# Commit
git add .
git commit -m "Phase 4: Configuration externalization

- Added PatronAnalyticsConfig class with all thresholds
- Documented reasoning for each parameter
- Added weight validation
- Updated functions to use config values
- Made parameters tunable without code changes

Validation: ✅ All outputs match baseline
Next: Phase 5 - Documentation"
````

---

## Phase 5: Documentation

**Goal**: Update docstrings and create usage docs
**Risk**: None
**Time**: 1 week
**Can Skip**: ✅ When time allows

(See full refactoring plan for details - this phase is mostly writing)

---

## Phase 6: Function Decomposition

**Goal**: Break down large functions
**Risk**: Medium
**Time**: 2-3 weeks
**Can Skip**: ✅ Only if have time and need

(This is optional advanced refactoring - see full plan for details)

---

## Testing Strategy

### After Every Change
````bash
# 1. Run pipeline
python MWSalesSumm.ipynb

# 2. Validate outputs
python validate_outputs.py

# 3. Check for errors in logs
grep -i "error\|exception\|failed" sales.log

# 4. If validation passes, commit
git add .
git commit -m "Phase X: Description..."
````

### Full Test Script

Save as `scripts/test_pipeline.py`:
````python
"""
Comprehensive testing after refactoring.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {description} PASSED")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ {description} FAILED")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        if result.stdout:
            print("Standard output:")
            print(result.stdout)
        return False

def full_test():
    """Run full test suite."""
    print("\n" + "="*60)
    print("MUSIC WORCESTER ANALYTICS - FULL TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Configuration validation
    all_passed &= run_command(
        "python -c 'from config import Config, PatronAnalyticsConfig; Config.validate(); PatronAnalyticsConfig.validate_weights()'",
        "Configuration Validation"
    )
    
    # Test 2: Import check
    all_passed &= run_command(
        "python -c 'import MW_functions; import Models_functions; print(\"Imports OK\")'",
        "Module Imports"
    )
    
    # Test 3: Run pipeline
    all_passed &= run_command(
        "python MWSalesSumm.ipynb",
        "Pipeline Execution"
    )
    
    # Test 4: Output validation
    all_passed &= run_command(
        "python validate_outputs.py",
        "Output Validation"
    )
    
    # Test 5: Check for errors in log
    if os.path.exists('sales.log'):
        with open('sales.log') as f:
            log_content = f.read()
            errors = [line for line in log_content.split('\n') 
                     if 'ERROR' in line or 'Exception' in line]
            if errors:
                print("\n⚠️  Errors found in sales.log:")
                for error in errors[:5]:  # Show first 5
                    print(f"  {error}")
                all_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(full_test())
````

---

## Rollback Procedures

### If Validation Fails
````bash
# 1. Don't panic! Check what changed
git status
git diff

# 2. Review the validation output
python validate_outputs.py

# 3. Check logs for clues
tail -n 50 sales.log

# 4. If unsure, rollback to last working commit
git log --oneline -5  # Find last good commit
git reset --hard <commit-hash>

# 5. Re-run validation
python validate_outputs.py

# 6. Should now pass
````

### If Pipeline Breaks
````bash
# 1. Check error message
python MWSalesSumm.ipynb 2>&1 | tee error.log

# 2. If import error, check file names
python -c "import MW_functions; import Models_functions"

# 3. If config error, check environment
python -c "from config import Config; Config.validate()"

# 4. If all else fails, rollback
git reset --hard HEAD~1

# 5. Investigate on a new branch
git checkout -b debug-issue
````

---

## Working with Claude (Memory Guide)

**Important**: Claude doesn't remember previous sessions. Each conversation starts fresh.

### Session Starter Template

Save as `docs/session_starter.md` and paste at start of each Claude session:
````markdown
# Session Context - Music Worcester Analytics Refactoring

## Project Overview
Patron analytics system for arts organization. Processes Salesforce ticket data
to generate RFM scores, segments, preferences, and retention metrics.

## Key Files
- MWSalesSumm.ipynb: Main notebook
- MW_functions.py: Data loading/cleaning
- Models_functions.py: Patron modeling
- config.py: Configuration
- validation.py: Input validation

## Refactoring Progress

### Completed ✅
- Phase 0: Git repo, baselines, validation script
- Phase 1: API keys externalized, imports fixed
- [Update as you progress]

### Current Phase 🔄
Phase [X]: [Description]

### Current Task
[Specific task you're working on]

## Question for This Session
[Your specific question]

## Relevant Code
[Paste only the specific function/module you're working on]
````

### Tips for Effective Sessions

1. **Be specific**: Don't ask "continue refactoring", ask "I'm on Phase 2, removing commented debug code. Review this cleanup script?"

2. **Share context**: Always paste the session starter + relevant code snippets

3. **One task per session**: Focus on completing one specific task

4. **Document as you go**: Update project_context.md after each session

5. **Use git commits as memory**: Detailed commit messages help you (and Claude) understand what's been done

### Recommended Workflow
````markdown
## Each Work Session

1. **Open project_context.md** - Review where you left off

2. **Check git log** - See recent changes
```bash
   git log --oneline -5
```

3. **Run validation** - Ensure starting from good state
```bash
   python validate_outputs.py
```

4. **Do work** - Make incremental changes

5. **Test immediately** - Don't accumulate changes
```bash
   python MWSalesSumm.ipynb
   python validate_outputs.py
```

6. **Commit frequently** - After each validation pass
```bash
   git add .
   git commit -m "Detailed description of what changed"
```

7. **Update docs** - Update project_context.md with progress

8. **If using Claude**:
   - Paste session_starter.md
   - Paste specific code you're working on
   - Ask focused question
````

---

## Summary Timeline

| Phase | Focus | Risk | Time | Can Skip? | Status |
|-------|-------|------|------|-----------|--------|
| 0 | Preparation | None | 2 hrs | ❌ Never | ⏳ |
| 1 | Security | Low | 1 wk | ❌ Never | ⏳ |
| 2 | Hygiene | Low | 1 wk | ⚠️ If pressed | ⏳ |
| 3 | Validation | Low | 1 wk | ⚠️ Recommended | ⏳ |
| 4 | Config | Low | 1 wk | ✅ Nice to have | ⏳ |
| 5 | Docs | None | 1 wk | ✅ When time allows | ⏳ |
| 6 | Decomposition | Med | 2-3 wks | ✅ Only if needed | ⏳ |

**Update this table as you complete each phase!**

---

## Key Principles

1. ✅ **Never skip Phase 0** - Always have a backup
2. ✅ **Always validate** - After every change
3. ✅ **Commit frequently** - After each passing validation
4. ✅ **One phase at a time** - Don't mix changes
5. ✅ **When in doubt, rollback** - Better safe than sorry
6. ✅ **Document everything** - Future you will thank you

---

## Files Created by This Refactoring
/Users/antho/Documents/WPI-MW/
├── .env                          # Your API keys (NOT in git)
├── .env.example                  # Template for API keys
├── .gitignore                    # Updated to protect secrets
├── config.py                     # Configuration management
├── validation.py                 # Input validation utilities
├── docs/
│   ├── refactoring_plan.md      # This document
│   ├── project_context.md       # Session starter
│   ├── refactoring_log.md       # Progress tracking
│   └── session_starter.md       # Claude session template
├── scripts/
│   ├── validate_outputs.py      # Output validation
│   ├── cleanup_script.py        # Remove commented code
│   └── test_pipeline.py         # Full test suite
└── *_baseline.csv               # Baseline outputs (not in git)
 
 
    validate_dataframe_not---

## Quick Reference Commands
````bash
# Validate configuration
python -c "from config import Config; Config.validate()"

# Run pipeline
python MWSalesSumm.ipynb

# Validate outputs
python validate_outputs.py

# Full test suite
python scripts/test_pipeline.py

# Check recent changes
git log --oneline -5
git diff

# Rollback if needed
git reset --hard HEAD~1

# Create new working branch
git checkout -b phase2-hygiene
````

---

## Support Resources

- **Original analysis**: See Claude conversation (this document)
- **Git history**: `git log` for detailed change history
- **Configuration docs**: See docstrings in `config.py`
- **Validation examples**: See `validation.py`

---

*End of Refactoring Guide*
*Last updated: 2024-01-12*
*Generated from Claude conversation*