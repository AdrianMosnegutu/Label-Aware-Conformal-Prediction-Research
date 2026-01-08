# Article Submission: Medical Image Analysis (Elsevier)

This directory contains the LaTeX source files for submission to **Medical Image Analysis**, an official journal of the MICCAI Society published by Elsevier.

---

## Journal Information

| Property | Details |
|----------|---------|
| **Journal** | Medical Image Analysis |
| **Publisher** | Elsevier |
| **ISSN** | 1361-8415 |
| **Impact Factor** | ~10.9 (2023) |
| **Society** | MICCAI Society (Official Journal) |
| **Review Type** | Single anonymized peer review |
| **Homepage** | https://www.sciencedirect.com/journal/medical-image-analysis |

### Scope & Aims

Medical Image Analysis publishes high-quality, original research in medical and biological image analysis, with emphasis on:

- Computer vision applications to biomedical imaging
- Algorithm development for medical image processing
- Machine learning and deep learning for medical imaging
- Segmentation, registration, and reconstruction
- Image-guided surgery and intervention
- Statistical shape analysis and computational anatomy

Supported imaging modalities include MRI, CT, X-ray, Ultrasound, PET/SPECT, Optical/Confocal Microscopy, and more.

---

## Document Class & Format

This manuscript uses the **elsarticle** document class, which is Elsevier's official LaTeX template.

### Current Configuration

```latex
\documentclass[final, 5p, times, twocolumn, authoryear]{elsarticle}
```

| Option | Description |
|--------|-------------|
| `final` | Camera-ready format (no draft marks) |
| `5p` | Two-column format matching published papers |
| `times` | Times font family |
| `twocolumn` | Two-column layout |
| `authoryear` | Author-year citation style (Harvard) |

### Switching Between Review and Final Formats

**For Submission (Reviewers prefer this):**
```latex
\documentclass[review, 12pt]{elsarticle}
```
- Single column, double-spaced
- Larger font for easier reading
- Enable `\linenumbers` for line numbering

**For Preview (Published paper appearance):**
```latex
\documentclass[final, 5p, times, twocolumn, authoryear]{elsarticle}
```
- Two-column layout
- Compact formatting
- Matches final published appearance

---

## Citation Style

The journal uses **Harvard (author-year)** citation style:

- Bibliography style: `elsarticle-harv`
- In-text citations: `\citep{key}` for (Author, Year) or `\citet{key}` for Author (Year)
- Requires `natbib` package

Example:
```latex
\citep{vovk2005algorithmic}  % → (Vovk et al., 2005)
\citet{vovk2005algorithmic}  % → Vovk et al. (2005)
```

---

## File Structure

```
article/
├── main.tex              # Main document (compile this)
├── main.pdf              # Compiled PDF output
├── references.bib        # BibTeX bibliography
├── highlights.txt        # Research highlights (3-5 bullet points)
├── README.md             # This file
└── sections/
    ├── abstract.tex      # Abstract (max 250 words)
    ├── keywords.tex      # Keywords (4-6 terms)
    ├── introduction.tex
    ├── related_work.tex
    ├── problem_formulation.tex
    ├── experimental_methodology.tex
    ├── conformal_prediction_methods.tex
    ├── results_and_analysis.tex
    ├── discussion.tex
    ├── conclusion.tex
    └── statements.tex    # CRediT, Data availability, Declarations
```

---

## Author Guidelines Summary

### Manuscript Requirements

| Element | Requirement |
|---------|-------------|
| **Abstract** | Max 250 words, structured summary |
| **Keywords** | 4-6 keywords, separated by semicolons |
| **Figures** | Min 300 dpi, TIFF/JPEG/EPS/PDF formats |
| **Tables** | Embedded in text, use `booktabs` for styling |
| **Equations** | Numbered, use `amsmath` environments |
| **References** | Complete entries, DOI preferred |

### Required Sections

1. **Highlights** (separate file): 3-5 bullet points, max 85 characters each
2. **Abstract**: Concise summary without citations
3. **Keywords**: Relevant index terms
4. **CRediT Author Statement**: Author contributions
5. **Declaration of Competing Interests**: Conflict disclosure
6. **Data Availability Statement**: Code/data access information

### Ethical Requirements

- Studies involving human subjects require ethics approval
- Patient data must be de-identified
- Informed consent documentation required
- Animal studies must follow institutional guidelines

---

## Building the PDF

From the project root directory:

```bash
./build_pdf.sh
```

Or manually:

```bash
cd article
latexmk -pdf main.tex
```

The build script automatically cleans auxiliary files after compilation.

---

## Submission Checklist

- [ ] Abstract within 250-word limit
- [ ] 4-6 keywords provided
- [ ] All figures high resolution (≥300 dpi)
- [ ] References complete with DOIs
- [ ] Highlights file (3-5 points, ≤85 chars each)
- [ ] CRediT author statement included
- [ ] Declaration of competing interests
- [ ] Data availability statement
- [ ] Ethics statement (if applicable)
- [ ] Line numbers enabled for review version
- [ ] Cover letter prepared

---

## Useful Links

- **Guide for Authors**: https://www.sciencedirect.com/journal/medical-image-analysis/publish/guide-for-authors
- **Submit Manuscript**: https://www.editorialmanager.com/media/
- **LaTeX Templates**: https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
- **Artwork Guidelines**: https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions

---

## Contact

For questions about the elsarticle class or Elsevier submission:
- Elsevier Support: https://service.elsevier.com/
- elsarticle documentation: `texdoc elsarticle`

