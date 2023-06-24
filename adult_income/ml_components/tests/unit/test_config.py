def test_config():
    from ml_components.config import RUNNER
    assert(RUNNER=='prod'),"RUNNER should already be 'prod' when pushing to BitBucket repo"