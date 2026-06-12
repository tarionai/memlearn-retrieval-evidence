# MANIFEST — file provenance and integrity

SHA-256 of every file in this repository, with its origin tag. Generated at assembly
time from the private origin tree; regenerate-and-diff to verify nothing drifted.

**Origin tags:**

- `byte-identical-to-origin` — copied verbatim from the private monorepo directory
  `apps/mem-learn/` (repository root maps to that directory; `packet/` maps to
  `retrieval-dashboard/public/packet/`). Hash equality with the origin file was
  verified at assembly time.
- `public-slice` — **the one modified code file.** `tournament_base.py` ships only the
  `bootstrap_ci` function (byte-identical in body) because the full private module
  imports the private CLI and its PostgreSQL dependency chain. Disclosed in the file's
  docstring.
- `curated-excerpt` — `tests/test_services_slice.py`: five test classes copied verbatim
  per class from the private `tests/unit/test_services.py`, whose remaining classes test
  the private memory kernel. Disclosed in the file's docstring.
- `authored-for-this-repo` — scaffolding written for this repository; no private origin.

The files under `packet/results/` additionally verify against the packet's own
`packet/results/SHA256SUMS.txt` (checked at assembly time). Internal work-package
references retained in docstrings are a deliberate publish decision — see README.

This manifest lists every tracked file except itself.

| File | Origin | SHA-256 |
|---|---|---|
| `.gitattributes` | authored-for-this-repo | `fe4cf7c3ac3734d0ef99976cf8015055fde9bc0b3766e3fc2d0c13793d4d4401` |
| `.github/workflows/tests.yml` | authored-for-this-repo | `41e5f38cb0c276f812b25bdb11b14191a71dbcd6aa6bbf4154d7d27dfef0f4ba` |
| `.gitignore` | authored-for-this-repo | `da65561049dbd51446a74d9d25444ed6c44a8879ea3b5edc04e1c2e26c45f4a0` |
| `LICENSE` | authored-for-this-repo | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `README.md` | authored-for-this-repo | `811a8befe7ef590c8ca9c27fb8e8f0526c01439ca1535d1a3daecc23494fca3e` |
| `benchmarks/__init__.py` | byte-identical-to-origin | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `benchmarks/retrieval_ablation/__init__.py` | byte-identical-to-origin | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `benchmarks/retrieval_ablation/metrics.py` | byte-identical-to-origin | `0e924cb73841ac8949e1a4e3dac28ac60c40bdc1bbe47efa4a927c8a9e65de21` |
| `benchmarks/retrieval_ablation/tournament_base.py` | public-slice | `08279d06f0bc74e63f475d15e30562a04bf7bebe5743423f8895a31a857c9dc5` |
| `packet/DATASET.md` | byte-identical-to-origin | `9df0e953aca07bd348ed86bed804f3f459c4c6f59aa9289291180127f59cf068` |
| `packet/ENVIRONMENT.md` | byte-identical-to-origin | `544e6e4eb6b441ba486b8840a2f64842fc293644182a35a50d8f94575b991c34` |
| `packet/EVAL_CONTRACT.md` | byte-identical-to-origin | `ce8ed78697c6546cb744eb900aaeffbbb55239cb3058c5367fbe8a4ba62fde09` |
| `packet/METRICS.md` | byte-identical-to-origin | `a722bc0a0a5f76cbf1ce7287ef22c0b8b9fe97951eb4ae5363147e6ca81d0749` |
| `packet/NEGATIVE_FINDINGS.md` | byte-identical-to-origin | `fca1da0f72a87df717969fddfef0d386d77d0a0c785b0ceec41c67ff61c1e760` |
| `packet/README.md` | byte-identical-to-origin | `91bf31c5ce1aee243cedbf9339e73801bfc0a3fda6f105b4a44b924281fcf2e1` |
| `packet/REPRODUCE.md` | byte-identical-to-origin | `aec0ff7e626dfebe73788853f91fd4d4aa4ecda2c910dc2723ab6e8828425a7f` |
| `packet/requirements.lock.txt` | byte-identical-to-origin | `d2d710aa848202fd16ca17135abcadb9f6e2c243448e92dc3c81bef6d3a2b65e` |
| `packet/results/SHA256SUMS.txt` | byte-identical-to-origin | `1636fba81a9f7cc0b0171251db7b8104de1ee2cd10c76da168933c4c0f6c247a` |
| `packet/results/hybrid_retrieval_findings_20260605.md` | byte-identical-to-origin | `2fee7ffe6c8f9bc88721a76e537fd0ad64e70f3795f1c5ba0aa6455b20b53fe7` |
| `packet/results/longmemeval_design_decisions.md` | byte-identical-to-origin | `2346265cd12e0bf1747be2462cba2c30f8c156f9f70e0f8befae137899ce7b98` |
| `packet/results/longmemeval_s_findings_20260604.md` | byte-identical-to-origin | `3ddfec5f193617cfc0ee7766f9b9b822544b1e686fe24556e78aa71aec6579af` |
| `packet/results/longmemeval_s_hybrid_results_n200.json` | byte-identical-to-origin | `97fb7eee6553c125953b9a6c04cfafc3d0c35a9c7cbc2e71c285105b30766e6b` |
| `packet/results/longmemeval_s_retrieval_results_n200.json` | byte-identical-to-origin | `633f58575a035cdecfae431aecb7ced27339eb151e44ebe9f18cfe14396cc217` |
| `packet/results/longmemeval_s_retrieval_results_n200_session.json` | byte-identical-to-origin | `4a705075ed8cec4ad34c7a1d70cd33914b679c84eb1fd4c0018caad4b3839690` |
| `packet/results/osam_selection_postmortem_20260604.md` | byte-identical-to-origin | `6d51d78f38402f8ac92c1b757775c105567298603e1c0658f301eac4275ea7ec` |
| `packet/results/ppr_arm_findings_20260604.md` | byte-identical-to-origin | `d8d2be8392136354c2f02ae74078efb61ba804e1dc986a65a82f6b657ceabc45` |
| `packet/results/ppr_arm_results_n200.json` | byte-identical-to-origin | `73e1481e6b56678bd8f2a5b59a2029870a260001b97ac77829ef76faf2acfe8e` |
| `pyproject.toml` | authored-for-this-repo | `6a25e7a47eb7e6e0fde33bd415e7ee63e51ae372018d8ffa88e085617b357965` |
| `src/memlearn/__init__.py` | byte-identical-to-origin | `d0cab45a830b4d7e87804ed6ab46ca3116194f889cb551a05f231da9f08ba49c` |
| `src/memlearn/adapters/__init__.py` | byte-identical-to-origin | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/memlearn/adapters/in_memory/__init__.py` | byte-identical-to-origin | `ec50b22ca594f91e017da0af3766516c8bea328bf9fbf4bb19617cde2112864c` |
| `src/memlearn/adapters/in_memory/fake_embedding.py` | byte-identical-to-origin | `2fffdbe63d7d4954e520f843fb32beddb757de819560d04ecfce180854e053f0` |
| `src/memlearn/adapters/in_memory/fake_entity_extractor.py` | byte-identical-to-origin | `6bd5a5857f14e1f015dede00f6379307354402538245bf069347963e1571ad40` |
| `src/memlearn/adapters/in_memory/fake_llm_port.py` | byte-identical-to-origin | `2dbc25d9a775950c991706c6e51fca40ca4736a55525c6b33071f0a201d0ad1d` |
| `src/memlearn/adapters/in_memory/fake_tokenizer.py` | byte-identical-to-origin | `c23e3c84cfd950d809b163dcf8b9cec3b5a9e5a8af26d6e0b4ad63f375960878` |
| `src/memlearn/adapters/in_memory/graph_store.py` | byte-identical-to-origin | `4857e86e15c60940175f9574346d1d4d5070c68304996cb87773338ce8ef6730` |
| `src/memlearn/adapters/in_memory/kv_store.py` | byte-identical-to-origin | `b465aa6135d42136e3b418fb9545e861ad706b5df3e6351354159e82c4165623` |
| `src/memlearn/adapters/in_memory/sentence_transformer_embedder.py` | byte-identical-to-origin | `4626903150e6fad56712d0d2f03b73abe3c2593c2e47a0a20c70a436f0b743ca` |
| `src/memlearn/adapters/in_memory/vector_store.py` | byte-identical-to-origin | `935f75bbcdedc2294813ef9dfc615b631610c35cfbbec05d24a33a7282649caa` |
| `src/memlearn/errors.py` | byte-identical-to-origin | `6b294adfde12929217c7b574267c294f9670c3d397b670254ee1b03f734f6d93` |
| `src/memlearn/ports.py` | byte-identical-to-origin | `4ae6b3f07ab86659fe28d655b4b7ec6a83efc6914196017b6a344663464e7c92` |
| `src/memlearn/primitives.py` | byte-identical-to-origin | `f4f61759aeac07d07726c478761ee942edde9d0384a2c0b32b0e04006c803a1d` |
| `src/memlearn/services/__init__.py` | byte-identical-to-origin | `a971fbd6335f24a2512ba098323517a317d583da43d3b2337003e01bd3cdad9b` |
| `src/memlearn/services/associative_state_engine.py` | byte-identical-to-origin | `6f4b696218e2ee43200ac16e92e752f27bb8b44a9e4d0ab78bf27b5b9fa57de1` |
| `src/memlearn/services/episodic_store.py` | byte-identical-to-origin | `1a5a5394212b2fbc9607641eef9a1967028600c21d11459a823b92bed0708bea` |
| `src/validation_mvp/__init__.py` | byte-identical-to-origin | `3d34355fb9e792879e3ad14172c8bee66a4034e49db8e0bdb75113ce6807754d` |
| `src/validation_mvp/bench_datasets/__init__.py` | byte-identical-to-origin | `42fbaaae09ea8d7704f5c53b1b8e9303a6a90199cb641b8ec555bf302086f164` |
| `src/validation_mvp/bench_datasets/longmemeval_adapter.py` | byte-identical-to-origin | `3047e08d3958ce53746de6d3607096bdd1a0e6a2d6b835c17e731e0ee2c24e28` |
| `src/validation_mvp/lexical_backends.py` | byte-identical-to-origin | `ea04845038e3620d25d6d49f6700d32c2e8a63e3217cdbd93a052a0276e3e7df` |
| `src/validation_mvp/ppr_arm.py` | byte-identical-to-origin | `c6fa9caa5b039fe69fdd7661ba98c9f99478105175898de38786c07a65c7b9e7` |
| `src/validation_mvp/run_longmemeval_retrieval.py` | byte-identical-to-origin | `84235f14856d95d31518347bc3890845dd472e7baee4e5b25b85c7aace33bb55` |
| `state/intermediate/.gitkeep` | authored-for-this-repo | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/__init__.py` | byte-identical-to-origin | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/test_ports.py` | byte-identical-to-origin | `9be067dfe02eaaf7d7b4f2e370f0343105939b3809e03abb3b3801f3e410868d` |
| `tests/test_primitives.py` | byte-identical-to-origin | `e36dd0da6eef1ea217feaba6db70552f810f05b39343c7124de049078b50bfe5` |
| `tests/test_services_slice.py` | curated-excerpt | `7b53ef6ce0bd36968b1a6cee2fa94ed41e9eae77cac3cceb2d6cc7a8b1c18f5a` |
| `tests/unit/__init__.py` | byte-identical-to-origin | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `tests/unit/test_in_memory_adapters.py` | byte-identical-to-origin | `5f88b3a60badfd8af11f266366864347f6a6c1312337393c0d11fe824f9e3210` |
