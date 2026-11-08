import re
from decimal import Decimal, InvalidOperation

from math_verify import parse, verify
from latex2sympy2_extended import latex2sympy
from sympy import simplify


def strip_thinking(text):
    text = str(text or "")

    # Some SFT checkpoints emit explicit Reasoning/Output wrappers.  Restrict
    # answer extraction to the final output block so numbers in the reasoning
    # cannot be mistaken for the answer.
    output_blocks = re.findall(
        r"<\|Output\|>(.*?)(?:<\|/Output\|>|$)", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if output_blocks:
        text = output_blocks[-1]
    else:
        text = re.sub(
            r"<\|Reasoning_step\|>.*?<\|/Reasoning_step\|>", "", text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if "<think>" in text and "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]

    text = re.sub(r"<\|/?(?:Output|Reasoning_step)\|>", "", text, flags=re.I)
    return text.strip()


def extract_balanced_boxed(text):
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        brace = text.find("{", start)
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1:index].strip()
    return ""


def extract_math_answer(text):
    content = strip_thinking(text)
    boxed = extract_balanced_boxed(content)
    if boxed:
        boxed_digits = re.sub(r"[^0-9]", "", boxed)
        base_values = re.findall(r"(?<![0-9])([0-9]+)\s*_\s*\{?([0-9]+)\}?", content)
        if boxed_digits and base_values:
            digits, base = base_values[-1]
            if digits == boxed_digits:
                return f"{digits}_{{{base}}}"
        return boxed
    patterns = (
        r"(?:therefore,?\s*)?(?:the\s+)?final answer(?:\s+is)?\s*:?\s*([^\n]+)",
        r"(?:the\s+)?answer is\s*:?\s*([^\n]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1]
            candidate = re.sub(r"\s*I hope it is correct\.?\s*$", "", candidate, flags=re.I)
            candidate = candidate.strip().strip("$").strip()
            # A leading dollar sign can be currency rather than a LaTeX delimiter.
            candidate = re.sub(r"^\$\s*", "", candidate)
            candidate = candidate.rstrip(".").strip()
            # A heading can itself be followed by a complete answer prefix.
            nested = re.search(
                r"(?:therefore,?\s*)?(?:the\s+)?final answer(?:\s+is)?\s*:?\s*(.+)$",
                candidate, flags=re.I,
            )
            if nested:
                candidate = nested.group(1).strip().rstrip(".")
            # Output blocks sometimes state the answer as a prose sentence.
            prose_tail = re.search(r"\bis\s*:?\s*(.+)$", candidate, flags=re.I)
            if prose_tail and len(re.findall(r"[A-Za-z]{3,}", candidate)) >= 3:
                candidate = prose_tail.group(1).strip().rstrip(".")
            return candidate
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    while lines and re.fullmatch(r"I hope it is correct\.?", lines[-1], flags=re.I):
        lines.pop()
    if lines:
        # Models often state the numeric result in prose without following the
        # requested boxed format. Prefer the last math/numeric expression.
        last = lines[-1]
        inline_math = re.findall(r"\\\((.+?)\\\)|\$(.+?)\$", last)
        if inline_math:
            groups = inline_math[-1]
            return next(value for value in groups if value).strip()
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", last)
        if numbers:
            return numbers[-1].replace(",", "")
    return lines[-1] if lines else content


def unwrap_command(text, command):
    pattern = re.compile(r"^\\" + command + r"\{(.*)\}$", re.DOTALL)
    match = pattern.match(text.strip())
    return match.group(1).strip() if match else text


def normalize_latex(text):
    text = str(text or "").strip().strip("$")
    text = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = re.sub(r"\\sqrt\s*([0-9A-Za-z])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])", r"\\frac{\1}{\2}", text)
    text = re.sub(r"_\{([0-9]+)\}", r"_\1", text)
    text = re.sub(r"\^\s*\{?\\circ\}?", "", text)
    text = re.sub(r"\\(?:text|mbox)\s*\{\s*(degrees?|degree|cents?|inches?|inch|cm|meters?|miles?|units?|square units?)\s*\}(?:\^\{?2\}?)?", "", text, flags=re.I)
    text = re.sub(r"\b(degrees?|degree|cents?|inches?|inch|cm|meters?|miles?|units?)\b(?:\^?2)?", "", text, flags=re.I)
    text = text.replace("\\$", "").replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\s+", "", text)
    return text.rstrip(".").strip()


def text_value(text):
    value = normalize_latex(text)
    for command in ("text", "mbox"):
        value = unwrap_command(value, command)
    value = value.strip("()").lower()
    return value


def split_top_level(text):
    parts, current, depth = [], [], 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def matrix_entries(text):
    match = re.search(r"\\begin\{(?:p|b|v)?matrix\}(.*?)\\end\{(?:p|b|v)?matrix\}", text, re.DOTALL)
    if not match:
        return None
    return [normalize_latex(x) for x in re.split(r"\\\\|&", match.group(1)) if x.strip()]


def symbolic_equal(left, right):
    if max(len(left), len(right)) > 300:
        return False
    prose_words = re.findall(r"[A-Za-z]{3,}", right.replace("\\text", "").replace("\\mbox", ""))
    if len(prose_words) >= 2:
        return False
    if re.search(r"_\s*\{?\d+\}?", left + right):
        return False
    try:
        left_expr = latex2sympy(left)
        right_expr = latex2sympy(right)
        if simplify(left_expr - right_expr) == 0:
            return True
    except Exception:
        pass
    try:
        gold = parse(left)
        pred = parse(right)
        return bool(gold and pred and verify(gold, pred, strict=False))
    except Exception:
        return False


def answer_equal(predicted, gold):
    pred = normalize_latex(predicted)
    target = normalize_latex(gold)
    if not pred or not target:
        return False, "empty"
    if pred.lower() == target.lower():
        return True, "normalized_exact"

    if text_value(pred) == text_value(target):
        return True, "text_or_unit"

    for lhs_prefix in (r"^[A-Za-z]+=", r"^[A-Za-z]+\\in"):
        stripped_target = re.sub(lhs_prefix, "", target)
        stripped_pred = re.sub(lhs_prefix, "", pred)
        if stripped_target != target or stripped_pred != pred:
            if stripped_target.lower() == stripped_pred.lower() or symbolic_equal(stripped_target, stripped_pred):
                return True, "equation_rhs"

    target_matrix, pred_matrix = matrix_entries(target), matrix_entries(pred)
    if target_matrix is not None or pred_matrix is not None:
        if target_matrix is None or pred_matrix is None or len(target_matrix) != len(pred_matrix):
            return False, "matrix_mismatch"
        if all(a == b or symbolic_equal(a, b) for a, b in zip(target_matrix, pred_matrix)):
            return True, "matrix"
        return False, "matrix_mismatch"

    target_parts, pred_parts = split_top_level(target), split_top_level(pred)
    if (len(target_parts) > 1 or len(pred_parts) > 1) and len(target_parts) != len(pred_parts):
        return False, "answer_count_mismatch"
    if len(target_parts) > 1 and len(target_parts) == len(pred_parts) and not target.startswith(("(", "[")):
        unmatched = pred_parts.copy()
        for item in target_parts:
            found = next((i for i, other in enumerate(unmatched) if item == other or symbolic_equal(item, other)), None)
            if found is None:
                break
            unmatched.pop(found)
        else:
            return True, "unordered_answers"

    try:
        if Decimal(pred) == Decimal(target):
            return True, "decimal"
    except InvalidOperation:
        pass

    if symbolic_equal(target, pred):
        return True, "symbolic"
    return False, "mismatch"
