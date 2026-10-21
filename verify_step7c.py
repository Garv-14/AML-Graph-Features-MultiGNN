import json
import re
import pandas as pd
from bs4 import BeautifulSoup

def parse_html_table(html_str):
    soup = BeautifulSoup(html_str, 'html.parser')
    table = soup.find('table')
    headers = [th.get_text().strip() for th in table.find('thead').find_all('th')]
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
        rows.append(cells)
    if rows and len(rows[0]) == len(headers) + 1:
        headers = ['index'] + headers
    df = pd.DataFrame(rows, columns=headers)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    return df

with open('notebooks/02_results.ipynb', 'r', encoding='utf-8') as f:
    nb02 = json.load(f)

# Find cell with ra.results_shap_summary()
shap_cell = None
for c in nb02['cells']:
    if c['cell_type'] == 'code' and 'ra.results_shap_summary()' in ''.join(c.get('source', [])):
        shap_cell = c
        break

assert shap_cell is not None, 'Could not find shap cell in 02_results.ipynb'

html_tables = []
for out in shap_cell.get('outputs', []):
    if 'data' in out and 'text/html' in out['data']:
        html = ''.join(out['data']['text/html'])
        html_tables.append(parse_html_table(html))

assert len(html_tables) == 2, f'Expected 2 HTML tables, found {len(html_tables)}'

df_nb_dir = html_tables[0]
df_nb_cat = html_tables[1]

print('=== DIRECTION TABLE (AS READ FROM NOTEBOOK) ===')
print(df_nb_dir.to_string(index=False))

print('\n=== CATEGORY TABLE (AS READ FROM NOTEBOOK) ===')
print(df_nb_cat.to_string(index=False))

# Load CSVs
df_csv_dir = pd.read_csv('results/direction_table.csv')
df_csv_cat = pd.read_csv('results/category_shap_corrected.csv')

# Assertions
assert len(df_nb_dir) == len(df_csv_dir), f'Direction row count mismatch: {len(df_nb_dir)} vs {len(df_csv_dir)}'
assert len(df_nb_cat) == len(df_csv_cat), f'Category row count mismatch: {len(df_nb_cat)} vs {len(df_csv_cat)}'

# Check direction table row by row
print('\nAsserting row-by-row equality with results/direction_table.csv ...')
for i in range(len(df_csv_dir)):
    nb_row = df_nb_dir.iloc[i]
    csv_row = df_csv_dir.iloc[i]
    assert str(nb_row['Rank']) == str(csv_row['Rank']), f'Row {i} Rank mismatch'
    assert str(nb_row['Feature']) == str(csv_row['Feature']), f'Row {i} Feature mismatch'
    assert float(nb_row['Mean |SHAP|']) == float(csv_row['Mean |SHAP|']), f'Row {i} Mean |SHAP| mismatch'
    assert str(nb_row['Spearman rho']) == str(csv_row['Spearman rho']), f'Row {i} rho mismatch'
    assert str(nb_row['Empirical Direction']) == str(csv_row['Empirical Direction']), f'Row {i} Direction mismatch'

print('=> Direction table row-by-row assertion: PASSED!')

# Check category table row by row
print('\nAsserting row-by-row equality with results/category_shap_corrected.csv ...')
for i in range(len(df_csv_cat)):
    nb_row = df_nb_cat.iloc[i]
    csv_row = df_csv_cat.iloc[i]
    assert str(nb_row['format name']) == str(csv_row['format name']), f'Row {i} format name mismatch'
    assert int(nb_row['code']) == int(csv_row['code']), f'Row {i} code mismatch'
    assert int(nb_row['sampled rows']) == int(csv_row['sampled rows']), f'Row {i} sampled rows mismatch'
    assert float(nb_row['mean SHAP']) == float(csv_row['mean SHAP']), f'Row {i} mean SHAP mismatch'
    assert float(nb_row['share of sampled positives']) == float(csv_row['share of sampled positives']), f'Row {i} share pos mismatch'
    assert float(nb_row['share of sampled negatives']) == float(csv_row['share of sampled negatives']), f'Row {i} share neg mismatch'
    assert int(nb_row['full-test rows']) == int(csv_row['full-test rows']), f'Row {i} full-test rows mismatch'
    assert int(nb_row['full-test positives']) == int(csv_row['full-test positives']), f'Row {i} full-test positives mismatch'

print('=> Category table row-by-row assertion: PASSED!')

# Search for strings 'Credit Card' next to '3.0484' or '92.43' in both notebooks
print('\nSearching both notebooks for Credit Card next to 3.0484 or 92.43 ...')
cc_mismatch_count = 0
for nb_file in ['notebooks/01_eda.ipynb', 'notebooks/02_results.ipynb']:
    with open(nb_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # Check within window of 100 characters
    for m in re.finditer(r'Credit Card', content):
        window = content[max(0, m.start()-100):min(len(content), m.end()+100)]
        if '3.0484' in window or '92.43' in window:
            print(f'Found suspicious proximity in {nb_file}: {window!r}')
            cc_mismatch_count += 1

print(f'Count of Credit Card next to 3.0484 or 92.43: {cc_mismatch_count}')
assert cc_mismatch_count == 0, f'Found {cc_mismatch_count} instances'
print('=> Proximity check: PASSED (count = 0)!')
