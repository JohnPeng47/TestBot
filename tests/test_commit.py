import pytest
from testbot.diff import CommitDiff

def test_contains_newtest_with_new_function():
    patch = """diff --git a/tests/test_example.py b/tests/test_example.py
index abc123..def456 100644
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -10,6 +10,10 @@ def test_existing():
     assert True
 
+def test_new_function():
+    assert True
+
 def another_test():
     assert False"""

    commit_diff = CommitDiff(patch)
    assert commit_diff.contains_newtest() == True

def test_contains_newtest_without_new_function():
    patch = """diff --git a/tests/test_example.py b/tests/test_example.py
index abc123..def456 100644
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -10,6 +10,7 @@ def test_existing():
     assert True
+    # Adding a comment
     assert False"""

    commit_diff = CommitDiff(patch)
    assert commit_diff.contains_newtest() == False
