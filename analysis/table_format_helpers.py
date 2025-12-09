#!/usr/bin/env python3
"""
Table formatting helper functions.

Separates data computation from presentation/formatting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TableMetadata:
    """
    Formatting metadata for a table (separate from data).
    
    Attributes:
        title: Table title for caption
        hypothesis: Which hypothesis this tests (e.g., "H1")
        footnotes: List of footnote strings
        number_formats: Dict mapping column name to format string (e.g., "{:.3f}")
        column_order: Preferred order of columns
        index_name: Name for index column (if applicable)
    """
    title: str
    hypothesis: str
    footnotes: List[str] = field(default_factory=list)
    number_formats: Dict[str, str] = field(default_factory=dict)
    column_order: List[str] = field(default_factory=list)
    index_name: Optional[str] = None
    
    
@dataclass  
class TablePackage:
    """
    Complete table package: data + formatting metadata.
    
    This separation allows:
    - Data computation to be independent of presentation
    - Multiple format outputs (LaTeX, CSV, markdown) from same data
    - Easy inspection of raw numbers
    """
    data: pd.DataFrame
    metadata: TableMetadata


# ============================================================================
# FORMATTING FUNCTIONS
# ============================================================================

def format_value(value, format_str):
    """Format a single value according to format string."""
    if pd.isna(value) or value is None:
        return "N/A"
    if np.isinf(value):
        return "∞"
    try:
        return format_str.format(value)
    except (ValueError, TypeError):
        return str(value)


def format_ci_column(df, base_col, ci_low_col, ci_high_col, format_str='{:.3f}'):
    """
    Combine base value with CI into formatted string.
    
    Args:
        df: DataFrame
        base_col: Column with base value
        ci_low_col: Column with CI lower bound
        ci_high_col: Column with CI upper bound
        format_str: Format string for numbers
        
    Returns:
        Series with formatted "value [ci_low, ci_high]" strings
    """
    result = []
    for _, row in df.iterrows():
        val = row[base_col]
        ci_low = row[ci_low_col]
        ci_high = row[ci_high_col]
        
        if pd.isna(val) or val is None:
            result.append("N/A")
        elif pd.isna(ci_low) or ci_low is None:
            # no CI available, just show value
            result.append(format_value(val, format_str))
        else:
            # show value with CI
            val_str = format_value(val, format_str)
            ci_low_str = format_value(ci_low, format_str)
            ci_high_str = format_value(ci_high, format_str)
            result.append(f"{val_str} [{ci_low_str}, {ci_high_str}]")
    
    return pd.Series(result, index=df.index)


def format_table_for_output(table_package: TablePackage, output_format='txt'):
    """
    Format a TablePackage for output in specified format.
    
    Args:
        table_package: TablePackage with data and metadata
        output_format: 'txt' (default), 'csv', or 'pickle' (raw data only)
        
    Returns:
        Formatted string ready for writing to file (or None for pickle)
    """
    df = table_package.data.copy()
    meta = table_package.metadata
    
    # identify columns that will be combined with CIs (don't format these yet)
    cols_with_ci = set()
    for col in df.columns:
        if col.endswith('_CI_low'):
            base_col = col[:-7]
            ci_high_col = f"{base_col}_CI_high"
            if base_col in df.columns and ci_high_col in df.columns:
                cols_with_ci.add(base_col)
    
    # apply number formatting only to columns WITHOUT CIs
    for col, fmt in meta.number_formats.items():
        if col in df.columns and col not in cols_with_ci:
            df[col] = df[col].apply(lambda x: format_value(x, fmt))
    
    # combine columns with CIs (these get formatted inside format_ci_column)
    cols_to_remove = []
    for col in df.columns:
        if col.endswith('_CI_low'):
            base_col = col[:-7]
            ci_high_col = f"{base_col}_CI_high"
            
            if base_col in df.columns and ci_high_col in df.columns:
                # get format string for this column
                fmt = meta.number_formats.get(base_col, '{:.3f}')
                
                # format_ci_column handles formatting of raw values
                df[base_col] = format_ci_column(df, base_col, col, ci_high_col, fmt)
                
                # mark CI columns for removal
                cols_to_remove.extend([col, ci_high_col])
    
    # remove separate CI columns
    df = df.drop(columns=cols_to_remove, errors='ignore')
    
    # remove SE columns (not needed in output)
    se_cols = [c for c in df.columns if c.endswith('_SE')]
    df = df.drop(columns=se_cols, errors='ignore')
    
    # reorder columns if specified
    if meta.column_order:
        # only use columns that actually exist
        cols = [c for c in meta.column_order if c in df.columns]
        # add any remaining columns not in order
        cols.extend([c for c in df.columns if c not in cols])
        df = df[cols]
    
    # set index if specified
    if meta.index_name and meta.index_name in df.columns:
        df = df.set_index(meta.index_name)
    
    # format based on output type
    if output_format == 'csv':
        return df.to_csv(index=True if meta.index_name else False)
    
    elif output_format == 'pickle':
        # for pickle, return None - caller will handle pickling the TablePackage
        return None
        
    else:  # txt (default) - clean plaintext
        output = []
        
        # title
        output.append("=" * 80)
        output.append(meta.title)
        output.append("=" * 80)
        output.append("")
        
        # hypothesis
        output.append(f"Hypothesis: {meta.hypothesis}")
        output.append("")
        
        # table using pandas string formatting
        table_str = df.to_string(index=True if meta.index_name else False)
        output.append(table_str)
        output.append("")
        
        # footnotes
        if meta.footnotes:
            output.append("Notes:")
            output.append("-" * 80)
            for i, note in enumerate(meta.footnotes, 1):
                # wrap long footnotes
                lines = _wrap_text(f"{i}. {note}", width=78)
                output.extend(lines)
            output.append("")
        
        return "\n".join(output)


def _wrap_text(text, width=78):
    """
    Wrap text to specified width while preserving words.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        
    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        # +1 for space before word (except first word)
        space_needed = word_length + (1 if current_line else 0)
        
        if current_length + space_needed <= width:
            current_line.append(word)
            current_length += space_needed
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else ['']