//! 对照 legacy 与 strict 两种步骤解析结果，用于 `step_parser = "strict"` 的推广排查。
//!
//! 为什么需要它：legacy 解析器会在两种常见写法下**静默丢掉步骤**——
//! ① 缩进的编号项（notes 子项）被当成步骤；② 步骤正文里贴的产物模板含 `## X`，
//! 被当成"新的二级标题"从而提前结束步骤段，后面的步骤（含人工审批门）直接不存在。
//! 丢掉的如果正好是人工审批门，引擎里就没有那道门了，而 SOP 文本上写着"不得代签"。
//!
//! 用法：
//!   cargo run --example sop_parse_diff -- <SOP 目录> [更多目录…]
//!   cargo run --example sop_parse_diff -- --json <SOP 目录>…
//!
//! 目录须含 SOP.toml 与 SOP.md。输出每个 SOP 在两种解析下的步骤数与人工门步号，
//! 以及当前 SOP.toml 里 step_parser 的取值。

use std::path::{Path, PathBuf};

fn gates(steps: &[zeroclaw::sop::types::SopStep]) -> Vec<u32> {
    steps
        .iter()
        .filter(|s| s.requires_confirmation)
        .map(|s| s.number)
        .collect()
}

fn titles(steps: &[zeroclaw::sop::types::SopStep]) -> Vec<String> {
    steps
        .iter()
        .map(|s| {
            let t: String = s.title.chars().take(60).collect();
            format!(
                "{}. {}{}",
                s.number,
                t,
                if s.requires_confirmation {
                    "  [人工门]"
                } else {
                    ""
                }
            )
        })
        .collect()
}

struct Row {
    dir: PathBuf,
    name: String,
    version: String,
    declared: String,
    legacy_n: usize,
    legacy_gates: Vec<u32>,
    legacy_titles: Vec<String>,
    strict_n: usize,
    strict_gates: Vec<u32>,
    strict_titles: Vec<String>,
}

impl Row {
    fn differs(&self) -> bool {
        self.legacy_n != self.strict_n || self.legacy_gates != self.strict_gates
    }
    /// 最严重的一类：legacy 下人工门为空，strict 下有——那道门在引擎里根本不存在。
    fn gate_lost(&self) -> bool {
        self.legacy_gates.is_empty() && !self.strict_gates.is_empty()
    }
}

fn scan(dir: &Path) -> Option<Row> {
    let toml_path = dir.join("SOP.toml");
    let md_path = dir.join("SOP.md");
    let toml_src = std::fs::read_to_string(&toml_path).ok()?;
    let md = std::fs::read_to_string(&md_path).unwrap_or_default();

    // 只取需要的三个字段，避免因 manifest 其它字段的 schema 变化而解析失败——
    // 这是排查工具，宁可少读也不要因为无关字段报错而漏掉一个 SOP。
    let v: toml::Value = toml::from_str(&toml_src).ok()?;
    let sop = v.get("sop")?;
    let field = |k: &str| {
        sop.get(k)
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string()
    };

    let legacy = zeroclaw::sop::parse_steps_with_mode(&md, false);
    let strict = zeroclaw::sop::parse_steps_with_mode(&md, true);

    Some(Row {
        dir: dir.to_path_buf(),
        name: field("name"),
        version: field("version"),
        declared: {
            let d = field("step_parser");
            if d.is_empty() {
                "(未声明→legacy)".to_string()
            } else {
                d
            }
        },
        legacy_n: legacy.len(),
        legacy_gates: gates(&legacy),
        legacy_titles: titles(&legacy),
        strict_n: strict.len(),
        strict_gates: gates(&strict),
        strict_titles: titles(&strict),
    })
}

fn fmt_gates(g: &[u32]) -> String {
    if g.is_empty() {
        "无".to_string()
    } else {
        g.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let dirs: Vec<PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();

    if dirs.is_empty() {
        eprintln!("用法: cargo run --example sop_parse_diff -- [--json] <SOP 目录>…");
        std::process::exit(2);
    }

    let rows: Vec<Row> = dirs.iter().filter_map(|d| scan(d)).collect();

    if json {
        let out: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "dir": r.dir,
                    "name": r.name,
                    "version": r.version,
                    "declared_step_parser": r.declared,
                    "legacy": {"steps": r.legacy_n, "gates": r.legacy_gates, "titles": r.legacy_titles},
                    "strict": {"steps": r.strict_n, "gates": r.strict_gates, "titles": r.strict_titles},
                    "differs": r.differs(),
                    "gate_lost_under_legacy": r.gate_lost(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return;
    }

    println!(
        "{:<30} {:<10} {:<16} {:>7} {:<10} {:>7} {:<10} 判定",
        "SOP", "版本", "声明", "legacy步", "legacy门", "strict步", "strict门"
    );
    for r in &rows {
        let verdict = if r.gate_lost() {
            "★ 人工门被丢弃"
        } else if r.differs() {
            "· 解析有差异"
        } else {
            "  一致"
        };
        println!(
            "{:<30} {:<10} {:<16} {:>7} {:<10} {:>7} {:<10} {}",
            r.name,
            r.version,
            r.declared,
            r.legacy_n,
            fmt_gates(&r.legacy_gates),
            r.strict_n,
            fmt_gates(&r.strict_gates),
            verdict
        );
    }

    let lost = rows.iter().filter(|r| r.gate_lost()).count();
    let diff = rows.iter().filter(|r| r.differs()).count();
    println!(
        "\n合计 {} 个 SOP：{} 个解析有差异，其中 {} 个的人工审批门在 legacy 下完全消失。",
        rows.len(),
        diff,
        lost
    );
}
