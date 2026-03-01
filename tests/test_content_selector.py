import pytest
from pdd.content_selector import ContentSelector, SelectorError

# Sample Python content for testing
PYTHON_SAMPLE = """
import os

def hello():
    "Docstring for hello."
    print("Hello")

@decorator
class MyClass:
    "Class docstring."
    
    def method_a(self):
        return 1
        
    def _private(self):
        pass

def world():
    return "World"
"""

# Sample Markdown content
MARKDOWN_SAMPLE = """
# Title

Intro text.

## Section 1
Content 1.

### Subsection 1.1
Content 1.1.

## Section 2
Content 2.
"""

def test_lines_selector():
    content = "L1\nL2\nL3\nL4\nL5"
    
    # Single line
    assert ContentSelector.select(content, "lines:1") == "L1"
    
    # Range
    assert ContentSelector.select(content, "lines:2-4") == "L2\nL3\nL4"
    
    # Start to M
    assert ContentSelector.select(content, "lines:-2") == "L1\nL2"
    
    # N to end
    assert ContentSelector.select(content, "lines:4-") == "L4\nL5"
    
    # Multiple ranges (comma separated)
    assert ContentSelector.select(content, "lines:1, lines:3, lines:5") == "L1\nL3\nL5"

    # Multiple ranges in one selector (passed as list to avoid splitting on comma)
    assert ContentSelector.select(content, ["lines:1,3,5"]) == "L1\nL3\nL5"

def test_lines_selector_out_of_bounds():
    content = "L1\nL2"
    with pytest.raises(SelectorError, match="Line 3 out of range"):
        ContentSelector.select(content, "lines:3")

def test_python_def_selector():
    res = ContentSelector.select(PYTHON_SAMPLE, "def:hello")
    assert 'def hello():' in res
    assert 'print("Hello")' in res
    assert 'class MyClass' not in res

    # Function not found
    with pytest.raises(SelectorError, match="Function 'missing' not found"):
        ContentSelector.select(PYTHON_SAMPLE, "def:missing")

def test_python_class_selector():
    res = ContentSelector.select(PYTHON_SAMPLE, "class:MyClass")
    assert 'class MyClass:' in res
    assert 'def method_a(self):' in res
    assert 'def hello():' not in res

def test_python_class_method_selector():
    res = ContentSelector.select(PYTHON_SAMPLE, "class:MyClass.method_a")
    assert 'def method_a(self):' in res
    assert 'class MyClass:' not in res # Should only return the method

    # Method not found
    with pytest.raises(SelectorError, match="Method 'missing' not found"):
        ContentSelector.select(PYTHON_SAMPLE, "class:MyClass.missing")

def test_markdown_section_selector():
    res = ContentSelector.select(MARKDOWN_SAMPLE, "section:Section 1", file_path="test.md")
    assert "## Section 1" in res
    assert "Content 1." in res
    assert "### Subsection 1.1" in res
    assert "## Section 2" not in res

    # Section not found
    with pytest.raises(SelectorError, match="Markdown section 'Missing' not found"):
        ContentSelector.select(MARKDOWN_SAMPLE, "section:Missing", file_path="test.md")

def test_pattern_selector():
    content = "apple\nbanana\ncherry"
    res = ContentSelector.select(content, "pattern:/nan/")
    assert res == "banana"
    
    # Pattern not found
    with pytest.raises(SelectorError, match="No lines matched pattern"):
        ContentSelector.select(content, "pattern:/xyz/")

def test_interface_mode_full_file():
    res = ContentSelector.select(PYTHON_SAMPLE, [], mode="interface")
    assert "def hello():" in res
    assert 'print("Hello")' not in res
    assert "..." in res
    assert "class MyClass:" in res
    assert "def method_a(self):" in res
    # Private members excluded
    assert "def _private(self):" not in res

def test_interface_mode_with_selector():
    # Only interface for specific function
    res = ContentSelector.select(PYTHON_SAMPLE, "def:hello", mode="interface")
    assert "def hello():" in res
    assert 'print("Hello")' not in res
    assert "..." in res
    assert "class MyClass" not in res

def test_multiple_selectors():
    content = "A\nB\nC\nD"
    # Select first and last line
    res = ContentSelector.select(content, ["lines:1", "lines:4"])
    assert res == "A\nD"

def test_malformed_selector():
    with pytest.raises(SelectorError, match="Malformed selector"):
        ContentSelector.select("content", "invalid:selector")

def test_file_type_enforcement():
    # AST selector on non-python file
    with pytest.raises(SelectorError, match="AST selectors require a .py file"):
        ContentSelector.select("content", "def:foo", file_path="test.txt")
        
    # Section selector on non-markdown file
    with pytest.raises(SelectorError, match="Section selector requires a .md file"):
        ContentSelector.select("content", "section:Foo", file_path="test.txt")

def test_duplicate_function_names():
    content = """
def foo():
    return 1

class Bar:
    def foo(self):
        return 2
"""
    res = ContentSelector.select(content, "def:foo")
    # Should include both top-level and method
    assert "def foo():" in res
    assert "return 1" in res
    assert "class Bar:" not in res
    assert "def foo(self):" in res
    assert "return 2" in res

def test_async_function():
    content = """
async def my_async():
    await something()
"""
    res = ContentSelector.select(content, "def:my_async")
    assert "async def my_async():" in res
    assert "await something()" in res

def test_nested_function():
    content = """
def outer():
    def inner():
        pass
"""
    res = ContentSelector.select(content, "def:inner")
    assert "def inner():" in res
    # It will extract just the inner function lines, indented as they are in the source
    assert "    def inner():" in res

