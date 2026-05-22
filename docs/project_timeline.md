# Project Timeline and Team Division

## Project Overview

- **Course**: CS 6476 Computer Vision (Georgia Tech)
- **Project**: Open-Vocabulary Object Detection and Visual Grounding
- **Model**: Grounding DINO
- **Dataset**: COCO 2017
- **Duration**: 6-8 weeks

---

## Team Members and Responsibilities

| Member | Role | Primary Responsibilities | Directory Ownership |
|--------|------|-------------------------|---------------------|
| **A** | Model Architect | Backbone, Language Encoder, Fusion Modules, Model Integration | `src/models/`, `configs/` |
| **B** | Data & Training Engineer | Dataset Loading, Data Augmentation, Training Pipeline, Training Scripts | `src/datasets/`, `src/engine/`, `scripts/` |
| **C** | Evaluation & Documentation | Evaluator, Visualization, Testing, Documentation, CI/CD | `src/inference/`, `src/utils/`, `tests/`, `docs/`, `notebooks/` |

---

## Detailed Timeline

### Phase 1: Environment Setup & Paper Study (Week 1-2)

**Goal**: Complete environment configuration and understand Grounding DINO architecture

| Task | Owner | Deliverable | Status |
|------|-------|-------------|--------|
| Set up development environment (PyTorch, CUDA, dependencies) | All | Working environment | ⬜ |
| Read Grounding DINO paper | All | Paper notes | ⬜ |
| Study GroundingDINO official codebase | All | Code understanding document | ⬜ |
| Download and prepare COCO dataset | B | Data ready | ⬜ |
| Set up project directory structure | A | Repository skeleton complete | ✅ |
| Configure pre-commit and CI | C | Code quality tools ready | ✅ |

**Milestone**: All team members can run the official GroundingDINO demo

---

### Phase 2: Model Reproduction - Core Modules (Week 3-4)

**Goal**: Implement core model components

| Task | Owner | Deliverable | Status |
|------|-------|-------------|--------|
| Implement Swin Transformer Backbone | A | `src/models/backbone/` | ⬜ |
| Implement BERT Language Encoder wrapper | A | `src/models/language/` | ⬜ |
| Implement Cross-Modal Fusion Module | A | `src/models/fusion/` | ⬜ |
| Implement COCO Dataset Loader | B | `src/datasets/coco_dataset.py` | ⬜ |
| Implement Data Augmentation Pipeline | B | `src/datasets/transforms.py` | ⬜ |
| Write unit tests | C | `tests/` directory | ⬜ |
| Set up training loop framework | B | `src/engine/trainer.py` | ⬜ |

**Milestone**: Individual modules pass unit tests

---

### Phase 3: Model Reproduction - Integration & Debugging (Week 5-6)

**Goal**: Complete full model and begin debugging

| Task | Owner | Deliverable | Status |
|------|-------|-------------|--------|
| Integrate modules into complete model | A | `src/models/grounding_dino.py` | ⬜ |
| Implement loss functions | A | Loss module | ⬜ |
| Implement evaluator (mAP calculation) | C | `src/engine/evaluator.py` | ⬜ |
| Write training script | B | `scripts/train.py` | ⬜ |
| Write evaluation script | C | `scripts/eval.py` | ⬜ |
| Begin small-scale debug training | All | Initial training results | ⬜ |

**Milestone**: Model can be trained on a small subset without errors

---

### Phase 4: Evaluation, Analysis & Report (Week 7-8)

**Goal**: Complete COCO evaluation and write final report

| Task | Owner | Deliverable | Status |
|------|-------|-------------|--------|
| Run full COCO evaluation | B | Evaluation results table | ⬜ |
| Result visualization and error analysis | C | Visualization plots | ⬜ |
| Ablation experiments (optional) | A | Comparison results | ⬜ |
| Write project report | All | Final report | ⬜ |
| Clean up code and documentation | C | Deliverable repository | ⬜ |
| Prepare presentation materials | A | PPT / Demo | ⬜ |

**Milestone**: Final submission ready

---

## Git Workflow

### Branch Strategy

```
main                        # Stable release (merge only completed features)
├── develop                 # Development mainline
│   ├── feat/backbone       # A: Swin Transformer backbone
│   ├── feat/language       # A: BERT language encoder
│   ├── feat/fusion         # A: Cross-modal fusion
│   ├── feat/dataset        # B: COCO dataset loader
│   ├── feat/trainer        # B: Training pipeline
│   ├── feat/evaluator      # C: Evaluation metrics
│   ├── feat/visualizer     # C: Result visualization
│   └── fix/xxx             # Bug fixes
```

### Commit Convention

```
feat(models): add Swin Transformer backbone
fix(datasets): correct bounding box format in COCO loader
docs(readme): update installation instructions
refactor(engine): simplify training loop
test(utils): add unit tests for box_ops
chore: update requirements.txt
```

---

## Communication

- **Weekly Meeting**: Every [Day] at [Time]
- **Communication Tool**: [Slack/Discord/WeChat]
- **Code Review**: At least one approval required before merging

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CUDA/GPU issues | Use Google Colab or university cluster as backup |
| Training too slow | Start with smaller model (Swin-T), reduce batch size |
| Dataset download issues | Pre-download data to shared drive |
| Merge conflicts | Regular pulls, small PRs, clear ownership |

---

## Resources

### Papers
- [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

### Code References
- [GroundingDINO Official](https://github.com/IDEA-Research/GroundingDINO)
- [COCO API](https://github.com/cocodataset/cocoapi)

### Datasets
- [COCO 2017](https://cocodataset.org/)
