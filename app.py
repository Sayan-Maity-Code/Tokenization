import streamlit as st
import unicodedata
import regex
from collections import Counter
from itertools import pairwise
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="Bengali BPE Tokenizer Visualizer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .challenge-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .grapheme-display {
        font-family: 'SolaimanLipi', 'Kalpurush', serif;
        font-size: 1.4rem;
        line-height: 2;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .token-display {
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        line-height: 1.8;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .byte-display {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #6c757d;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .grapheme-analysis {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    .normalization-demo {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #9c27b0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def analyze_bengali_text_structure(text):
    """Analyze Bengali text structure showing graphemes, normalization, and encoding"""
    
    # Original text (before normalization)
    original_text = text
    
    # Normalize using NFC
    normalized_text = unicodedata.normalize("NFC", text)
    
    # Extract grapheme clusters using \X
    graphemes = regex.findall(r'\X', normalized_text)
    
    # Analyze each grapheme
    grapheme_analysis = []
    for i, grapheme in enumerate(graphemes):
        # Get Unicode code points
        code_points = [ord(c) for c in grapheme]
        code_point_names = [unicodedata.name(c, f"U+{ord(c):04X}") for c in grapheme]
        
        # Get UTF-8 bytes
        utf8_bytes = list(grapheme.encode('utf-8'))
        
        # Check if it's a complex grapheme (multiple code points)
        is_complex = len(code_points) > 1
        
        grapheme_info = {
            'index': i,
            'grapheme': grapheme,
            'code_points': code_points,
            'code_point_names': code_point_names,
            'utf8_bytes': utf8_bytes,
            'byte_count': len(utf8_bytes),
            'is_complex': is_complex,
            'complexity_type': classify_bengali_complexity(grapheme)
        }
        grapheme_analysis.append(grapheme_info)
    
    return {
        'original_text': original_text,
        'normalized_text': normalized_text,
        'graphemes': graphemes,
        'grapheme_analysis': grapheme_analysis,
        'normalization_changed': original_text != normalized_text
    }

def classify_bengali_complexity(grapheme):
    """Classify the type of Bengali complexity in a grapheme"""
    code_points = [ord(c) for c in grapheme]
    
    if len(code_points) == 1:
        return "Simple character"
    
    # Check for common Bengali combining patterns
    complexity_types = []
    
    # Vowel diacritics (মাত্রা)
    bengali_vowel_signs = range(0x09BE, 0x09C5)  # Various vowel signs
    if any(cp in bengali_vowel_signs for cp in code_points):
        complexity_types.append("Vowel diacritic")
    
    # Consonant conjuncts (যুক্তাক্ষর)
    bengali_virama = 0x09CD  # Hasanta
    if bengali_virama in code_points:
        complexity_types.append("Consonant conjunct")
    
    # Other combining marks
    if any(unicodedata.category(chr(cp)) == 'Mn' for cp in code_points):
        complexity_types.append("Combining mark")
    
    return " + ".join(complexity_types) if complexity_types else "Complex cluster"

def bengali_utf8_bpe_with_grapheme_awareness(text, max_merges=10):
    """Enhanced BPE that respects grapheme boundaries"""
    
    # Step 1: Analyze text structure
    text_analysis = analyze_bengali_text_structure(text)
    normalized_text = text_analysis['normalized_text']
    graphemes = text_analysis['graphemes']
    
    # Step 2: Convert each grapheme to bytes (maintaining grapheme integrity)
    grapheme_byte_sequences = []
    flat_sequence = []
    
    for grapheme in graphemes:
        bytes_for_grapheme = list(grapheme.encode("utf-8"))
        grapheme_byte_sequences.append(bytes_for_grapheme)
        flat_sequence.extend(bytes_for_grapheme)
    
    initial_length = len(flat_sequence)
    next_token_id = 256
    merges = {}
    merge_history = []
    
    # Step 3: BPE merging process
    sequence = flat_sequence.copy()
    
    for merge_num in range(1, max_merges + 1):
        pair_counts = Counter(pairwise(sequence))
        if not pair_counts:
            break
            
        (most_common_pair, freq) = pair_counts.most_common(1)[0]
        if freq < 2:
            break
        
        # Record merge details with grapheme context
        merge_info = {
            'merge_num': merge_num,
            'pair': most_common_pair,
            'frequency': freq,
            'token_id': next_token_id,
            'sequence_length_before': len(sequence),
            'represents_grapheme_boundary': check_grapheme_boundary(most_common_pair, grapheme_byte_sequences)
        }
        
        merges[next_token_id] = most_common_pair
        
        # Apply merge
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and (sequence[i], sequence[i+1]) == most_common_pair:
                new_sequence.append(next_token_id)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        
        sequence = new_sequence
        merge_info['sequence_length_after'] = len(sequence)
        merge_history.append(merge_info)
        next_token_id += 1
    
    return sequence, merges, merge_history, initial_length, text_analysis

def check_grapheme_boundary(byte_pair, grapheme_byte_sequences):
    """Check if a byte pair represents a meaningful grapheme boundary"""
    for seq in grapheme_byte_sequences:
        if len(seq) >= 2:
            for i in range(len(seq) - 1):
                if (seq[i], seq[i+1]) == byte_pair:
                    return True
    return False

def generate_grapheme_aware_visualization(text_analysis, compressed_tokens, merges):
    """Generate visualization that shows grapheme clusters properly"""
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
              '#F8C471', '#82E0AA', '#AED6F1', '#F1C40F', '#E74C3C']
    
    def expand_token(token_id):
        if token_id < 256:
            return [token_id]
        else:
            left, right = merges[token_id]
            return expand_token(left) + expand_token(right)
    
    # Create mapping from byte position to grapheme
    byte_to_grapheme = {}
    current_byte_pos = 0
    
    for g_idx, grapheme in enumerate(text_analysis['graphemes']):
        grapheme_bytes = list(grapheme.encode('utf-8'))
        for byte_offset in range(len(grapheme_bytes)):
            byte_to_grapheme[current_byte_pos + byte_offset] = {
                'grapheme_idx': g_idx,
                'grapheme': grapheme,
                'is_start': byte_offset == 0,
                'is_end': byte_offset == len(grapheme_bytes) - 1
            }
        current_byte_pos += len(grapheme_bytes)
    
    # Generate colored output
    colored_html = ""
    current_pos = 0
    token_colors = {}
    color_idx = 0
    
    for token in compressed_tokens:
        if token not in token_colors:
            if token >= 256:
                token_colors[token] = colors[color_idx % len(colors)]
                color_idx += 1
            else:
                token_colors[token] = '#666666'
        
        expanded_bytes = expand_token(token)
        
        # Check if this token spans multiple graphemes
        graphemes_in_token = set()
        for byte_pos in range(current_pos, current_pos + len(expanded_bytes)):
            if byte_pos in byte_to_grapheme:
                graphemes_in_token.add(byte_to_grapheme[byte_pos]['grapheme_idx'])
        
        # Reconstruct the character(s) for this token
        try:
            token_text = bytes(expanded_bytes).decode('utf-8', errors='ignore')
            
            # Add border if token spans grapheme boundaries
            border_style = ""
            if len(graphemes_in_token) > 1:
                border_style = "border: 2px dashed #FF0000; "
            
            if token_text.strip():
                colored_html += f'<span style="background-color: {token_colors[token]}; {border_style}padding: 2px 4px; margin: 1px; border-radius: 3px; color: white; font-weight: bold;" title="Token ID: {token}, Spans {len(graphemes_in_token)} grapheme(s)">{token_text}</span>'
            else:
                colored_html += token_text
        except:
            colored_html += f'<span style="background-color: {token_colors[token]}; padding: 2px 4px; margin: 1px; border-radius: 3px; color: white; font-weight: bold;" title="Token ID: {token}">[?]</span>'
        
        current_pos += len(expanded_bytes)
    
    return colored_html, token_colors

def generate_grapheme_boundary_visualization(text_analysis, compressed_tokens, merge_table):
    """Generate visualization showing grapheme boundaries"""
    
    def expand_token(token_id):
        if token_id < 256:
            return [token_id]
        else:
            left, right = merge_table[token_id]
            return expand_token(left) + expand_token(right)
    
    # Create boundary markers
    boundary_html = ""
    current_pos = 0
    
    for token in compressed_tokens:
        expanded_bytes = expand_token(token)
        
        try:
            token_text = bytes(expanded_bytes).decode('utf-8', errors='ignore')
            boundary_html += f'<span style="background-color: #e0e7ff; padding: 2px 4px; margin: 1px; border-radius: 3px; border: 1px solid #3b82f6;">{token_text}</span>'
        except:
            boundary_html += f'<span style="background-color: #fee2e2; padding: 2px 4px; margin: 1px; border-radius: 3px; border: 1px solid #ef4444;">[?]</span>'
    
    return boundary_html

def bpe_decode(token_sequence, merges, errors="strict"):
    """Decode BPE tokens back to text"""
    def expand_token(token_id):
        if token_id < 256:
            return [token_id]
        else:
            left, right = merges[token_id]
            return expand_token(left) + expand_token(right)
    
    byte_ids = []
    for tid in token_sequence:
        byte_ids.extend(expand_token(tid))
    
    raw_bytes = bytes(byte_ids)
    return raw_bytes.decode("utf-8", errors=errors)

# App Header
st.markdown('<div class="main-header">🔤 Bengali BPE Tokenizer with Grapheme Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h3>🎯 Understanding Bengali Script Complexity</h3>
Bengali script uses <strong>grapheme clusters</strong> - what your eyes see as "one character" might be composed of multiple Unicode code points. 
This creates unique challenges for tokenization that don't exist in simple scripts like English.
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced educational content
with st.sidebar:
    st.header("📚 Bengali Script Challenges")
    
    st.markdown("""
    ### 🔍 Grapheme Clusters:
    - **Visual Unit**: What you see as one character
    - **Technical Reality**: Multiple Unicode code points
    - **Example**: কি = ক (ka) + ি (i vowel sign)
    - **Regex \\X**: Captures complete grapheme clusters
    """)
    
    st.markdown("""
    ### 🔄 Normalization (NFC):
    - **Problem**: Same visual character, different encoding
    - **Solution**: Unicode normalization
    - **NFC**: Canonical composed form
    - **Critical**: Must normalize before tokenizing
    """)
    
    st.markdown("""
    ### ⚡ Tokenization Impact:
    - **Grapheme Splitting**: Breaks visual characters
    - **Context Loss**: Meaning depends on complete cluster
    - **Inefficiency**: More tokens for same meaning
    - **Model Confusion**: Partial characters are meaningless
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input Bengali Text")
    
    # Sample texts showing different complexities
    sample_options = {
        "Simple Text": "আমি বাংলায় কথা বলি।",
        "Complex Conjuncts": "ক্ষুদ্র ক্রিয়া ব্যক্তিত্ব স্বাধীনতা",
        "Mixed Diacritics": "কী হয়েছে? তুমি কোথায় গিয়েছিলে?",
        "Long Speech": """সুভাষ চন্দ্র বসু জয়ন্তীর ভাষণ
অধ্যক্ষ মহোদয়, সম্মানিত শিক্ষকগণ এবং এখানে উপস্থিত আমার প্রিয় বন্ধুরা। তোমাদের সকলকে আমার শুভেচ্ছা।"""
    }
    
    selected_sample = st.selectbox("Choose a sample text:", list(sample_options.keys()))
    
    user_text = st.text_area(
        "Enter Bengali text to analyze:",
        value=sample_options[selected_sample],
        height=150,
        help="Try different types of Bengali text to see how grapheme complexity affects tokenization"
    )
    
    max_merges = st.slider(
        "Number of BPE Merges:",
        min_value=1,
        max_value=15,
        value=8,
        help="More merges = better compression but may break grapheme boundaries"
    )

with col2:
    st.header("🔬 Grapheme Analysis")
    
    if user_text.strip():
        # Analyze text structure
        text_analysis = analyze_bengali_text_structure(user_text)
        
        # Show normalization effects
        if text_analysis['normalization_changed']:
            st.markdown("""
            <div class="normalization-demo">
            <h4>🔄 Normalization Applied</h4>
            <p>Text was normalized (NFC) - internal representation changed while appearance remained same.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display graphemes with analysis
        st.markdown("**Grapheme Clusters (what you see as characters):**")
        
        # Initialize grapheme display
        grapheme_display = ""

        for i, grapheme in enumerate(text_analysis['graphemes']):
            if grapheme.strip():  # Skip whitespace-only graphemes
                analysis = text_analysis['grapheme_analysis'][i]
                complexity_color = "#FF6B6B" if analysis['is_complex'] else "#4ECDC4"

                byte_count = analysis['byte_count']
                complexity_type = analysis['complexity_type']

                grapheme_display += (
                    f'<span style="background-color: {complexity_color}; '
                    f'padding: 4px 6px; margin: 2px; border-radius: 4px; '
                    f'color: white; font-weight: bold;" '
                    f'title="Bytes: {byte_count}, Type: {complexity_type}">'
                    f'{grapheme}</span>'
                )
            else:
                grapheme_display += grapheme

        
        st.markdown(f'<div class="grapheme-display">{grapheme_display}</div>', unsafe_allow_html=True)
        st.caption("🔴 Complex graphemes (multiple code points) | 🟢 Simple characters")

# Detailed Analysis Section
if user_text.strip():
    st.header("🔍 Detailed Bengali Structure Analysis")
    
    # Get analysis
    text_analysis = analyze_bengali_text_structure(user_text)
    
    # Grapheme breakdown table
    st.subheader("📊 Grapheme Breakdown")
    
    grapheme_data = []
    for analysis in text_analysis['grapheme_analysis']:
        if analysis['grapheme'].strip():  # Skip whitespace
            grapheme_data.append({
                'Grapheme': analysis['grapheme'],
                'Code Points': len(analysis['code_points']),
                'UTF-8 Bytes': analysis['byte_count'],
                'Complexity': analysis['complexity_type'],
                'Unicode Names': ' + '.join([name.split()[-1] if 'BENGALI' in name else name for name in analysis['code_point_names']])[:50] + "..."
            })
    
    if grapheme_data:
        df_graphemes = pd.DataFrame(grapheme_data)
        st.dataframe(df_graphemes, use_container_width=True)
        
        # Complexity statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_graphemes = len([g for g in text_analysis['graphemes'] if g.strip()])
        complex_graphemes = len([g for g in text_analysis['grapheme_analysis'] if g['is_complex'] and g['grapheme'].strip()])
        total_bytes = sum([g['byte_count'] for g in text_analysis['grapheme_analysis'] if g['grapheme'].strip()])
        avg_bytes_per_grapheme = total_bytes / total_graphemes if total_graphemes > 0 else 0
        
        with col1:
            st.metric("Total Graphemes", total_graphemes)
        with col2:
            st.metric("Complex Clusters", complex_graphemes)
        with col3:
            st.metric("Total UTF-8 Bytes", total_bytes)
        with col4:
            st.metric("Avg Bytes/Grapheme", f"{avg_bytes_per_grapheme:.1f}")

# BPE Tokenization with Grapheme Awareness
if user_text.strip():
    st.header("🔬 BPE Tokenization Analysis")
    
    with st.spinner("Performing grapheme-aware BPE tokenization..."):
        compressed_tokens, merge_table, merge_history, initial_length, text_analysis = bengali_utf8_bpe_with_grapheme_awareness(user_text, max_merges)
    
    # Create two columns for different visualizations
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.subheader("🎨 Token-wise Visualization")
        st.caption("Each token has a unique color. Red dashed border = splits graphemes")
        
        # Generate enhanced visualization
        colored_tokens, token_colors = generate_grapheme_aware_visualization(text_analysis, compressed_tokens, merge_table)
        st.markdown(f'<div class="token-display">{colored_tokens}</div>', unsafe_allow_html=True)
        
        # Show token legend
        with st.expander("🗂️ Token Color Legend"):
            legend_html = ""
            for i, token in enumerate(compressed_tokens[:20]):  # Show first 20 tokens
                if token in token_colors:
                    try:
                        def expand_token(token_id):
                            if token_id < 256:
                                return [token_id]
                            else:
                                left, right = merge_table[token_id]
                                return expand_token(left) + expand_token(right)
                        
                        expanded_bytes = expand_token(token)
                        token_text = bytes(expanded_bytes).decode('utf-8', errors='ignore')
                        legend_html += (
                            f'<span style="background-color: {token_colors[token]}; padding: 2px 6px; '
                            f'margin: 2px; border-radius: 3px; color: white; font-weight: bold; '
                            f'display: inline-block;">Token {i+1}: "{token_text}" (ID: {token})</span><br>'
                        )
                    except:
                        legend_html += (
                            f'<span style="background-color: {token_colors[token]}; padding: 2px 6px; '
                            f'margin: 2px; border-radius: 3px; color: white; font-weight: bold; '
                            f'display: inline-block;">Token {i+1}: [?] (ID: {token})</span><br>'
                        )
            if len(compressed_tokens) > 20:
                legend_html += f"<p>... and {len(compressed_tokens) - 20} more tokens</p>"
            st.markdown(legend_html, unsafe_allow_html=True)
    
    with vis_col2:
        st.subheader("🔍 Grapheme Boundary Analysis")
        st.caption("Shows how tokens align with grapheme clusters")
        
        # Generate grapheme boundary visualization
        boundary_vis = generate_grapheme_boundary_visualization(text_analysis, compressed_tokens, merge_table)
        st.markdown(f'<div class="token-display">{boundary_vis}</div>', unsafe_allow_html=True)
    
    # Show token IDs in full width
    st.markdown("**🔢 Token ID Sequence:**")
    token_display = " ".join([f"[{token}]" for token in compressed_tokens])
    if len(token_display) > 1000:  # Truncate if too long
        truncated = " ".join([f"[{token}]" for token in compressed_tokens[:50]])
        token_display = truncated + f" ... (showing first 50 of {len(compressed_tokens)} tokens)"
    st.markdown(f'<div class="byte-display">{token_display}</div>', unsafe_allow_html=True)
    
    # Analysis of grapheme boundary violations
    st.subheader("⚠️ Grapheme Boundary Analysis")
    
    # Count violations more accurately
    violations = 0
    violation_details = []
    
    current_pos = 0
    for i, token in enumerate(compressed_tokens):
        def expand_token(token_id):
            if token_id < 256:
                return [token_id]
            else:
                left, right = merge_table[token_id]
                return expand_token(left) + expand_token(right)
        
        expanded_bytes = expand_token(token)
        
        # Check if this token spans multiple graphemes
        graphemes_in_token = set()
        for byte_pos in range(current_pos, current_pos + len(expanded_bytes)):
            # Map byte position to grapheme
            byte_pos_in_text = 0
            for g_idx, grapheme in enumerate(text_analysis['graphemes']):
                grapheme_bytes = list(grapheme.encode('utf-8'))
                if byte_pos_in_text <= byte_pos < byte_pos_in_text + len(grapheme_bytes):
                    graphemes_in_token.add(g_idx)
                    break
                byte_pos_in_text += len(grapheme_bytes)
        
        if len(graphemes_in_token) > 1:
            violations += 1
            try:
                token_text = bytes(expanded_bytes).decode('utf-8', errors='ignore')
                violation_details.append(f"Token {i+1}: '{token_text}' spans {len(graphemes_in_token)} graphemes")
            except:
                violation_details.append(f"Token {i+1}: [ID:{token}] spans {len(graphemes_in_token)} graphemes")
        
        current_pos += len(expanded_bytes)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tokens", len(compressed_tokens))
    with col2:
        st.metric("Grapheme Violations", violations)
    with col3:
        violation_rate = (violations / len(compressed_tokens) * 100) if len(compressed_tokens) > 0 else 0
        st.metric("Violation Rate", f"{violation_rate:.1f}%")
    with col4:
        compression_ratio = ((initial_length - len(compressed_tokens)) / initial_length * 100) if initial_length > 0 else 0
        st.metric("Compression", f"{compression_ratio:.1f}%")
    
    if violations > 0:
        st.markdown("""
        <div class="challenge-box">
        <h4>⚠️ Grapheme Boundary Issues Detected</h4>
        <p>Some tokens split grapheme clusters, which can harm model understanding of Bengali text. 
        This is why Bengali needs specialized tokenization approaches.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Detailed Violation List"):
            for detail in violation_details[:10]:  # Show first 10 violations
                st.write(f"• {detail}")
            if len(violation_details) > 10:
                st.write(f"... and {len(violation_details) - 10} more violations")
    
    else:
        st.markdown("""
        <div class="success-box">
        <h4>✅ No Grapheme Boundary Violations</h4>
        <p>All tokens respect grapheme cluster boundaries! This is ideal for Bengali text processing.</p>
        </div>
        """, unsafe_allow_html=True)

# Educational Comparison
st.header("📚 Why Grapheme Awareness Matters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="challenge-box">
    <h4>❌ Problems with Naive Tokenization</h4>
    <ul>
    <li><strong>Splits Visual Characters:</strong> কি becomes ক + ি tokens</li>
    <li><strong>Loses Meaning:</strong> Partial graphemes are meaningless</li>
    <li><strong>Inefficient:</strong> More tokens needed for same text</li>
    <li><strong>Model Confusion:</strong> AI can't learn proper Bengali patterns</li>
    <li><strong>Poor Generation:</strong> May produce invalid character combinations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="success-box">
    <h4>✅ Grapheme-Aware Solutions</h4>
    <ul>
    <li><strong>Preserve Visual Units:</strong> Keep grapheme clusters intact</li>
    <li><strong>Meaningful Tokens:</strong> Each token represents complete concepts</li>
    <li><strong>Better Compression:</strong> Fewer tokens for same semantic content</li>
    <li><strong>Improved Learning:</strong> AI learns proper Bengali patterns</li>
    <li><strong>Valid Output:</strong> Generated text follows script rules</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
# Technical Deep Dive
with st.expander("🔬 Technical Implementation Details"):
    st.markdown("""
    ### Key Technical Approaches:
    
    1. **Unicode Normalization (NFC)**:
       ```python
       normalized = unicodedata.normalize("NFC", text)
       ```
       Ensures consistent internal representation.
    
    2. **Grapheme Cluster Extraction**:
       ```python
       graphemes = regex.findall(r'\\X', normalized_text)
       ```
       Uses `\\X` pattern to capture complete visual characters.
    
    3. **Grapheme-Aware BPE**:
       - Track which byte pairs cross grapheme boundaries
       - Penalize or avoid merges that split visual characters
       - Prioritize merges within grapheme clusters
    
    4. **Visualization Enhancement**:
       - Color-code tokens by merge level
       - Highlight grapheme boundary violations
       - Show byte-to-grapheme mapping
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
<h4>🎓 Key Insights for Bengali Tokenization</h4>
<ol>
<li><strong>Grapheme clusters are the true units</strong> of Bengali text that must be preserved</li>
<li><strong>Unicode normalization is essential</strong> before any text processing</li>
<li><strong>Standard BPE often breaks Bengali characters</strong> leading to poor model performance</li>
<li><strong>Specialized tokenizers are needed</strong> for complex scripts like Bengali</li>
<li><strong>Visual representation helps understand</strong> why regional languages struggle with current AI models</li>
</ol>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #6c757d;">
🇧🇩 Built to highlight the importance of script-aware tokenization for Bengali and other complex writing systems
</div>
""", unsafe_allow_html=True)