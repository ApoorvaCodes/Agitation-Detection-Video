# Agitation Detection Video Research Pipeline

This repository contains a staged, research-oriented pipeline for video evidence
review. Person 1 extracts person tracks, pose and motion; Person 2 generates
candidate behaviour intervals; Person 3 builds compact evidence, validates
candidates, and presents traceable results. None of these stages establishes a
diagnosis or clinical validity.

Architecture:

`Video → P1 perception → P2 candidate generation → P3 evidence packet → CMAI-aware quality checks → optional Qwen/Groq verification → deterministic Python event timestamps → evidence clips/dashboard/optional Supabase`


## Person 1: Video Perception

Person 1 extracts person tracks, pose landmarks, normalized body representations,
and motion features from video.

Install with:

```sh
pip install -e '.[models,test]'