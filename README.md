# MAK2+: Democratizing Access to Precision Molecular Diagnostics

**Replace expensive ddPCR hardware ($60,000+) with free, open-source software for absolute quantification of nucleic acids.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)]()
[![Paper: In Preparation](https://img.shields.io/badge/Paper-In%20Preparation-blue.svg)]()

---

## 🌍 The Problem

Accurate absolute quantification of pathogens is critical for:
- Malaria surveillance and elimination efforts
- Clinical trial endpoints (drug efficacy, vaccine studies)
- Treatment monitoring (viral load, parasite density)

**Current solution:** Droplet Digital PCR (ddPCR)
- 💰 Cost: $60,000-100,000 per instrument
- 💸 Plus $2-5 per reaction in consumables
- 🚫 **Result:** Unaffordable for 85% of laboratories in malaria-endemic countries

**The impact:** Most labs in resource-limited settings cannot access gold-standard diagnostics, limiting disease surveillance and research capacity exactly where it's needed most.

---

## ✨ The Solution: MAK2+

**MAK2+ provides ddPCR-equivalent absolute quantification using:**
- ✅ Your existing qPCR machine (no new hardware!)
- ✅ Standard qPCR reagents and protocols
- ✅ Free, open-source computational analysis
- ✅ No standard curves required

**Cost:** $0 (vs $60,000+)  
**Time to deploy:** Download and use immediately  
**Accuracy:** Validation study in progress

---

## 🚀 Quick Start

### Install locally:
```bash
# Clone repository
git clone https://github.com/gboggy2/MAK2-plus.git
cd MAK2-plus

# Install dependencies
pip install -r requirements.txt

# Run web interface
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📊 How It Works

MAK2+ uses **mechanistic kinetic modeling** of the PCR amplification process:

1. **Upload your qPCR data** (CSV, Excel, or manual entry)
2. **Fit the MAK2 model** to your amplification curves
3. **Get absolute quantification** without standard curves

**Under the hood:** The MAK2 model (Boggy & Woolf, 2010) describes PCR chemistry using mass-action kinetics, naturally accounting for declining amplification efficiency and primer depletion.

**Result:** Direct estimation of initial template concentration from a single qPCR run.

👉 [Technical Details](TECHNICAL_README.md) | Publication in preparation

---

## 🎯 Use Cases

### 🔬 Current Focus:
- **Malaria parasitemia** (Plasmodium spp.) - Validation study in progress

### 🌟 Potential Applications:
- TB quantification (Mycobacterium tuberculosis)
- Viral load monitoring (HIV, SARS-CoV-2)
- Copy number variation
- Gene expression (absolute)
- GMO detection
- Environmental samples (pathogen surveillance)

**Using MAK2+ in your research?** We'd love to hear about it! [Open an issue](https://github.com/gboggy2/MAK2-plus/issues) or email gjboggy@gmail.com

---

## 📈 Validation & Performance

**Head-to-head validation study ongoing:**
- Comparing MAK2+ to ddPCR for Plasmodium parasitemia quantification
- Testing across 10¹-10⁷ parasite density range  
- Multiple species and sample types (cultured + clinical)
- Collaboration with malaria research laboratory
- Results expected early 2025

**Status:**
- Methodology validated against published MAK2 model (Boggy & Woolf, 2010)
- Multi-site validation in progress
- Manuscript in preparation

**Using MAK2+ in your lab?** We're actively seeking validation partners. Contact gjboggy@gmail.com to collaborate.

---

## 💡 Why Open Source?

We believe precision diagnostics should be accessible to **every laboratory**, regardless of location or resources.

**By making MAK2+ open source:**
- 🌍 Labs in low-resource settings can access ddPCR-quality quantification
- 🔬 Researchers can verify and improve the methods
- 📚 Scientists can cite validated, peer-reviewed techniques
- 🤝 Community contributions accelerate innovation

**This is science for global health equity.**

---

## 📖 Documentation

- **[Technical Details](TECHNICAL_README.md)** - Model equations and implementation
- **Quick Start Guide** - Coming soon
- **User Manual** - Coming soon
- **API Reference** - Coming soon
- **FAQ** - Coming soon

---

## 🤝 Contributing

We welcome contributions! Ways to help:

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/gboggy2/MAK2-plus/issues)
- 💡 **Suggest features** for your use case
- 📝 **Improve documentation** (especially for non-English speakers)
- 🧪 **Share validation data** from your lab
- 💻 **Contribute code** via Pull Requests

---

## 📜 Citation

If you use MAK2+ in your research, please cite:

**Software:**
```
Boggy, G. (2024). MAK2+: Open-source computational tool for absolute 
nucleic acid quantification. GitHub. https://github.com/gboggy2/MAK2-plus
```

**Validation paper** (when published):
```
[Citation will be added upon publication]
```

**Original MAK2 model:**
```
Boggy, G.J., & Woolf, P.J. (2010). A Mechanistic Model of PCR for 
Accurate Quantification of Quantitative PCR Data. PLOS ONE, 5(8), e12355.
https://doi.org/10.1371/journal.pone.0012355
```

---

## 🏆 Recognition & Support

This project is part of the **Open Diagnostics Initiative** (nonprofit in formation), working to democratize access to molecular diagnostics globally.

**Current status:** Beta (actively seeking validation partners and feedback)

**Future funding:** Grant applications in preparation to:
- Validate across multiple pathogen types
- Deploy in malaria-endemic countries
- Provide training and support
- Expand to TB, HIV, and other critical diagnostics

**Interested in partnering?** Contact gjboggy@gmail.com

---

## 📄 License

MAK2+ is released under the [MIT License](LICENSE).

**What this means:**
- ✅ Free to use for any purpose (academic, commercial, personal)
- ✅ Free to modify and distribute
- ✅ No restrictions on derivative works
- ⚠️ Provided "as is" without warranty (see license for details)

---

## 📞 Contact & Support

- **Issues & Bug Reports:** [GitHub Issues](https://github.com/gboggy2/MAK2-plus/issues)
- **General Questions:** [GitHub Discussions](https://github.com/gboggy2/MAK2-plus/discussions)
- **Email:** gjboggy@gmail.com

**For collaboration inquiries** (validation studies, partnerships, funding):  
Email gjboggy@gmail.com with subject "MAK2+ Collaboration"

---

## 🙏 Acknowledgments

- **Original MAK2 model:** Boggy & Woolf (2010)
- **Community contributors:** List coming soon
- **Built with:** Python, Streamlit, SciPy, NumPy, Plotly

**Developed by:** Gregory Boggy, Ph.D.  
**Organization:** Open Diagnostics Initiative (nonprofit in formation)

---

## 🗺️ Roadmap

**Current (December 2024 - March 2025):**
- ✅ Core MAK2+ implementation complete
- ✅ Web interface deployed
- 🔄 ddPCR validation study (seeking partners)
- 🔄 Manuscript preparation

**Next (Q2-Q3 2025):**
- 📊 Bootstrap confidence intervals
- 🌐 Multi-language support (French, Spanish, Portuguese)
- 🧪 Batch processing improvements
- 📱 Mobile interface optimization

**Vision (2025-2026):**
- 🌍 Global deployment in endemic countries
- 🔬 Expanded pathogen validation (TB, HIV)
- 🤝 Partnerships with global health organizations
- 💰 Sustainable nonprofit model established

---

**⭐ Star this repo** if MAK2+ helps your research!  
**📣 Share** with colleagues who need affordable absolute quantification!

---

*Making precision diagnostics accessible to every laboratory, everywhere.* 🌍
