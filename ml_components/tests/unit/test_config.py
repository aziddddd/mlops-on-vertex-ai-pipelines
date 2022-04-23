def test_config():
    from kfp_components.config import RUNNER
    assert(RUNNER=='prod'),"RUNNER should already be 'prod' when pushing to BitBucket repo"