// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

use markdown::mdast::{AttributeContent, AttributeValue, Node};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Mirror the `/STACK` setting from `.cargo/config.toml`, which is not shipped
/// in the published crate.
fn bump_windows_stack_size() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:8000000");
    }
}

const STD_WIDGETS_DOCS: &str = "std-widgets-docs";
const NON_WIDGET_PAGES: &[&str] = &["overview.mdx", "style.mdx"];
/// The nested widgets which appear as level two headers instead of their own pages.
/// Any changes to decls or to the documentation in this regard should be caught by
/// tests
const CHILD_WIDGET_SECTIONS: &[(&str, &str)] =
    &[("RadioButton", "RadioButton"), ("Properties of the `Tab` element", "Tab")];

/// parsed from <SlintProperty> in the mdx
struct SlintProperty {
    name: String,
    type_name: String,
    visibility: String,
    default: Option<String>,
}

/// Generate rust code to look up a widget's documentation by name. The generated file is
/// std_widget_docs.rs
fn generate_std_widget_docs_module() {
    println!("cargo:rerun-if-changed={STD_WIDGETS_DOCS}");
    let mut docs = BTreeMap::new();
    let mut properties = BTreeMap::<(String, Option<String>), Vec<SlintProperty>>::new();
    for path in gather_mdx_files_recursive(Path::new(STD_WIDGETS_DOCS)) {
        let file_name = path.file_name().unwrap().to_str().unwrap();
        if NON_WIDGET_PAGES.contains(&file_name) {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let (title, markdown, page_properties) = mdx_to_markdown(&path, &text);
        if let Some(previous) = docs.insert(title.clone(), (path.clone(), markdown)) {
            panic!("{} and {} both have the title {title}", previous.0.display(), path.display());
        }
        for (child, property) in page_properties {
            properties.entry((title.clone(), child)).or_default().push(property);
        }
    }

    let mut code = String::from(
        "/// The markdown documentation of the std-widgets widget `name`.\n\
         pub fn std_widget_doc(name: &str) -> Option<&'static str> {\n    match name {\n",
    );
    for (title, (_, markdown)) in &docs {
        writeln!(code, "        {title:?} => Some({markdown:?}),").unwrap();
    }
    code.push_str("        _ => None,\n    }\n}\n\n");
    code.push_str(
        "/// The properties that the reference pages document, by widget and,\n\
         /// for a child element such as `Tab` of `TabWidget`, child element name,\n\
         /// as `(name, type, visibility)`.\n\
         #[cfg(test)]\n\
         pub static STD_WIDGET_PROPERTIES: &[(&str, Option<&str>, &[(&str, &str, &str)])] = &[\n",
    );
    for ((widget, child), properties) in &properties {
        write!(code, "    ({widget:?}, {child:?}, &[").unwrap();
        for p in properties {
            write!(code, "({:?}, {:?}, {:?}), ", p.name, p.type_name, p.visibility).unwrap();
        }
        code.push_str("]),\n");
    }
    code.push_str("];\n");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("std_widget_docs.rs");
    std::fs::write(out, code).unwrap();
}

fn gather_mdx_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(gather_mdx_files_recursive(&path));
        } else if path.extension().is_some_and(|ext| ext == "mdx") {
            files.push(path);
        }
    }
    files.sort();
    files
}

/// Converts a reference page to plain markdown,
/// returning its `title`, the markdown, and the properties it documents,
/// each with the child element it belongs to, if any.
/// Panics on markup it doesn't know how to convert, so the build fails.
fn mdx_to_markdown(
    path: &Path,
    text: &str,
) -> (String, String, Vec<(Option<String>, SlintProperty)>) {
    let text = admonitions_to_jsx(text);
    let tree = parse_mdx(path, &text);
    let mut converter =
        MdxConverter { path, source: &text, title: None, child: None, properties: Vec::new() };
    let markdown = converter.blocks(tree.children().unwrap());
    let Some(title) = converter.title.take() else {
        converter.fail(&tree, "no title in the front matter")
    };
    (title, markdown, converter.properties)
}

fn parse_mdx(path: &Path, text: &str) -> Node {
    let mut options = markdown::ParseOptions::mdx();
    options.constructs.frontmatter = true;
    // Accept any `import` statement without parsing the JavaScript.
    options.mdx_esm_parse = Some(Box::new(|_| markdown::MdxSignal::Ok));
    markdown::to_mdast(text, &options).unwrap_or_else(|e| panic!("{}:{e}", path.display()))
}

/// Replaces the `:::kind[label]` and `:::` lines of Starlight admonitions with
/// `<Admonition kind="kind" label="label">` and `</Admonition>`,
/// which the MDX parser reads as JSX.
/// The markdown parser would read these lines as paragraph text.
/// Assumes no code block contains such a line.
fn admonitions_to_jsx(text: &str) -> String {
    let mut result = String::new();
    for line in text.lines() {
        match line.trim().strip_prefix(":::") {
            None => result.push_str(line),
            Some("") => result.push_str("</Admonition>"),
            Some(kind) => match kind.split_once('[') {
                Some((kind, label)) => {
                    let label = label.strip_suffix(']').unwrap_or(label);
                    write!(result, "<Admonition kind=\"{kind}\" label=\"{label}\">").unwrap()
                }
                None => write!(result, "<Admonition kind=\"{kind}\">").unwrap(),
            },
        }
        result.push('\n');
    }
    result
}

struct MdxConverter<'a> {
    path: &'a Path,
    source: &'a str,
    title: Option<String>,
    /// The child element that the current section documents, such as `Tab` of `TabWidget`.
    child: Option<String>,
    properties: Vec<(Option<String>, SlintProperty)>,
}

impl MdxConverter<'_> {
    fn fail(&self, node: &Node, msg: &str) -> ! {
        let line = node.position().map_or(0, |p| p.start.line);
        panic!("{}:{line}: {msg}", self.path.display())
    }

    fn source_of(&self, nodes: &[Node]) -> &str {
        match (nodes.first().and_then(Node::position), nodes.last().and_then(Node::position)) {
            (Some(first), Some(last)) => &self.source[first.start.offset..last.end.offset],
            _ => "",
        }
    }

    /// The markdown of `nodes`, separated by blank lines.
    fn blocks(&mut self, nodes: &[Node]) -> String {
        let blocks = nodes.iter().map(|node| self.block(node)).collect::<Vec<_>>();
        blocks.into_iter().filter(|b| !b.is_empty()).collect::<Vec<_>>().join("\n\n")
    }

    fn block(&mut self, node: &Node) -> String {
        match node {
            Node::Yaml(yaml) => {
                self.title = yaml
                    .value
                    .lines()
                    .find_map(|line| line.strip_prefix("title:"))
                    .map(|title| title.trim().to_string());
                String::new()
            }
            Node::MdxjsEsm(_) => String::new(),
            Node::MdxFlowExpression(e) if is_comment(&e.value) => String::new(),
            // Drops the docs-site attributes after the language, such as `playground`.
            Node::Code(code) => {
                format!("```{}\n{}\n```", code.lang.as_deref().unwrap_or(""), code.value)
            }
            Node::Heading(heading) if heading.depth == 2 => {
                let text = self.source_of(&heading.children);
                self.child = CHILD_WIDGET_SECTIONS
                    .iter()
                    .find(|(h, _)| *h == text)
                    .map(|(_, child)| child.to_string());
                self.with_inline_jsx(node)
            }
            Node::MdxJsxFlowElement(element) => {
                let attributes = self.attributes(node, &element.attributes);
                match element.name.as_deref() {
                    Some("CodeSnippetMD") => self.blocks(&element.children),
                    Some("SlintProperty") => {
                        let p = property(&attributes).unwrap_or_else(|e| self.fail(node, &e));
                        let default = match &p.default {
                            Some(default) => format!(" (default: `{default}`)"),
                            None => String::new(),
                        };
                        let signature = format!(
                            "`{} property <{}> {}`{default}",
                            p.visibility, p.type_name, p.name
                        );
                        self.properties.push((self.child.clone(), p));
                        let body = self.blocks(&element.children);
                        if body.is_empty() { signature } else { format!("{signature}\n\n{body}") }
                    }
                    Some("Admonition") => {
                        let label =
                            attributes.get("label").map(|l| l.to_string()).unwrap_or_else(|| {
                                let mut chars = attributes["kind"].chars();
                                chars
                                    .next()
                                    .map(|c| c.to_uppercase().chain(chars).collect())
                                    .unwrap_or_default()
                            });
                        let body = self.blocks(&element.children);
                        let quoted = body.lines().map(|line| {
                            if line.is_empty() { ">".into() } else { format!("> {line}") }
                        });
                        std::iter::once(format!("> **{label}**\n>"))
                            .chain(quoted)
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                    name => self.fail(node, &format!("unsupported tag '<{}>'", name.unwrap_or(""))),
                }
            }
            _ => self.with_inline_jsx(node),
        }
    }

    /// The source of `node` with each inline `<Link>` replaced by its text.
    fn with_inline_jsx(&self, node: &Node) -> String {
        let mut links = Vec::new();
        self.collect_links(node, &mut links);
        let position = node.position().unwrap();
        let mut result = String::new();
        let mut copied = position.start.offset;
        for (start, end, replacement) in links {
            result.push_str(&self.source[copied..start]);
            result.push_str(&replacement);
            copied = end;
        }
        result.push_str(&self.source[copied..position.end.offset]);
        result
    }

    fn collect_links(&self, node: &Node, links: &mut Vec<(usize, usize, String)>) {
        match node {
            Node::MdxJsxTextElement(element) if element.name.as_deref() == Some("Link") => {
                let attributes = self.attributes(node, &element.attributes);
                let link = link(&attributes).unwrap_or_else(|e| self.fail(node, &e));
                let position = node.position().unwrap();
                links.push((position.start.offset, position.end.offset, link));
            }
            Node::MdxJsxTextElement(markdown::mdast::MdxJsxTextElement { name, .. })
            | Node::MdxJsxFlowElement(markdown::mdast::MdxJsxFlowElement { name, .. }) => {
                self.fail(node, &format!("unsupported tag '<{}>'", name.as_deref().unwrap_or("")))
            }
            Node::MdxTextExpression(e) if is_comment(&e.value) => {
                let position = node.position().unwrap();
                links.push((position.start.offset, position.end.offset, String::new()));
            }
            Node::MdxTextExpression(_) | Node::MdxFlowExpression(_) => {
                self.fail(node, "unsupported expression")
            }
            _ => {
                for child in node.children().into_iter().flatten() {
                    self.collect_links(child, links);
                }
            }
        }
    }

    fn attributes<'n>(
        &self,
        node: &Node,
        attributes: &'n [AttributeContent],
    ) -> BTreeMap<&'n str, &'n str> {
        attributes
            .iter()
            .map(|attribute| match attribute {
                AttributeContent::Property(p) => {
                    let value = match &p.value {
                        Some(AttributeValue::Literal(value)) => value.as_str(),
                        Some(AttributeValue::Expression(expression)) => expression.value.as_str(),
                        None => "",
                    };
                    (p.name.as_str(), value)
                }
                AttributeContent::Expression(_) => {
                    self.fail(node, "unsupported attribute expression")
                }
            })
            .collect()
    }
}

/// The property that the attributes of a `<SlintProperty>` tag describe.
fn property(attributes: &BTreeMap<&str, &str>) -> Result<SlintProperty, String> {
    const KNOWN: &[&str] = &[
        "propName",
        "typeName",
        "defaultValue",
        "propertyVisibility",
        "enumName",
        "structName",
        "inlineReferenceDocs",
    ];
    if let Some(unknown) = attributes.keys().find(|k| !KNOWN.contains(k)) {
        return Err(format!("unsupported SlintProperty attribute '{unknown}'"));
    }
    let name = attributes.get("propName").ok_or("SlintProperty without propName")?;
    let mut type_name =
        attributes.get("typeName").ok_or("SlintProperty without typeName")?.to_string();
    for (placeholder, attribute) in [("struct", "structName"), ("enum", "enumName")] {
        match (type_name.contains(placeholder), attributes.get(attribute)) {
            (true, Some(actual)) => type_name = type_name.replace(placeholder, actual),
            (true, None) => return Err(format!("typeName '{type_name}' without {attribute}")),
            (false, Some(_)) => return Err(format!("{attribute} on typeName '{type_name}'")),
            (false, None) => {}
        }
    }
    Ok(SlintProperty {
        name: name.to_string(),
        type_name,
        visibility: attributes.get("propertyVisibility").unwrap_or(&"in").to_string(),
        default: attributes.get("defaultValue").map(|d| d.to_string()),
    })
}

/// Whether an MDX expression such as `{/* cSpell: ignore */}` holds only a comment.
fn is_comment(expression: &str) -> bool {
    let expression = expression.trim();
    expression.starts_with("/*") && expression.ends_with("*/")
}

/// The text of a `<Link type="X" />`: its `label`, or `` `X` `` when it has none.
fn link(attributes: &BTreeMap<&str, &str>) -> Result<String, String> {
    if let Some(unknown) = attributes.keys().find(|k| !["type", "label"].contains(k)) {
        return Err(format!("unsupported Link attribute '{unknown}'"));
    }
    match (attributes.get("label"), attributes.get("type")) {
        (Some(label), _) => Ok(label.to_string()),
        (None, Some(ty)) => Ok(format!("`{ty}`")),
        (None, None) => Err("Link without type or label".into()),
    }
}

fn main() {
    #[cfg(feature = "preview-engine")]
    slint_build::compile(".").unwrap();
    bump_windows_stack_size();
    generate_std_widget_docs_module();
}
