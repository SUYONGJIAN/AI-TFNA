# \# AI-TFNA: AI-Thyroid Fine Needle Aspiration Cytological Classification

# 

# > Multimodal deep learning for thyroid nodule cytological diagnosis aligned with The Bethesda System (TBS).

# 

# ---

# 

# \## 🔗 Quick Links

# \- Paper: \[Cytological classification Diagnosis for thyroid nodules via multimodal model deep learning](<insert DOI/arXiv link here when available>)

# \- GitHub: https://github.com/SUYONGJIAN/AI-TFNA

# 

# ---

# 

# \## 📌 Overview

# AI-TFNA is a multimodal deep-learning pipeline that automatically classifies thyroid fine-needle aspiration (FNA) cytology slides according to The Bethesda System (TBS).  

# The framework integrates nuclear morphology, cellular phenotype and slide-level context to deliver highly accurate cytological diagnosis and BRAF-mutation prediction.

# 

# ---

# 

# \## 🏗️ Architecture

# | Module | Role |

# |--------|------|

# | \*\*SEG-DETECT\*\* | Nuclear segmentation \& morphological feature extraction (XFPN-U-Net backbone) |

# | \*\*VAN-tiny\*\*   | Single-cell / cluster-cell phenotype classification |

# | \*\*XGBoost\*\*    | Slide-level TBS category prediction |

# 

