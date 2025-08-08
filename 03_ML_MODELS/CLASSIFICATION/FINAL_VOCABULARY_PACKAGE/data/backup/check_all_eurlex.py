#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# Φόρτωση και εξέταση του νέου αρχείου
data = json.load(open('eurovoc_all_eurlex_concepts.json', encoding='utf-8'))
print(f'📊 Συνολικά concepts: {len(data)}')

# Ανάλυση τύπων concepts
concepts_with_alt_labels = sum(1 for v in data.values() if v.get('alt_labels'))
concepts_with_paths = sum(1 for v in data.values() if 'hierarchical_paths' in v)
concepts_with_parents = sum(1 for v in data.values() if v.get('parents'))

print(f'🏷️  Concepts με alt_labels: {concepts_with_alt_labels}')
print(f'🌳 Concepts με parents: {concepts_with_parents}')
print(f'📚 Concepts με paths: {concepts_with_paths}')

# Δείγματα
print('\n📋 ΔΕΙΓΜΑΤΑ CONCEPTS:')
examples = list(data.items())[:5]
for i, (concept_id, concept_data) in enumerate(examples):
    print(f'\n{i+1}. ID: {concept_id}')
    print(f'   Title: {concept_data.get("title", "N/A")}')
    print(f'   Alt labels: {len(concept_data.get("alt_labels", []))} labels')
    print(f'   Has paths: {"hierarchical_paths" in concept_data}')
    print(f'   Has parents: {"parents" in concept_data}')
