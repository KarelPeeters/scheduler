use std::collections::HashSet;
use std::fmt::Write;

use itertools::{enumerate, Itertools};
use rouille::{Request, Response};

use crate::core::expand::expand;
use crate::core::problem::Problem;
use crate::core::schedule::{Action, ActionChannel, ActionDrop, Timed};
use crate::core::state::State;
use crate::core::wrapper::TypedIndex;

pub fn main_server(problem: Problem) {
    rouille::start_server("localhost:8000", move |request| {
        handle_request(&problem, request)
    });
}

fn handle_request(problem: &Problem, request: &Request) -> Response {
    let url = request.url();
    let path = url.strip_prefix("/").unwrap()
        .trim_end_matches("/")
        .split_terminator("/").collect_vec();
    println!("{} {:?}", url, path);

    let r = match path.as_slice() {
        &[] => {
            Response::redirect_303("actions/")
        }
        &["actions", ref indices @ ..] => {
            let indices: Vec<u64> = match indices.iter().map(|s| s.parse()).try_collect() {
                Ok(indices) => indices,
                Err(_) => return Response::empty_404(),
            };
            let (parent, state, children) = match pick_state(problem, &indices) {
                Ok(state) => state,
                Err(_) => return Response::empty_404(),
            };

            Response::html(build_html(problem, &indices, parent.as_ref(), &state, &children))
        }
        _ => Response::empty_404(),
    };

    println!("{:?}", r);

    r
}

fn pick_state(problem: &Problem, indices: &[u64]) -> Result<(Option<State>, State, Vec<State>), ()> {
    let mut prev = None;
    let mut curr = State::new(problem);

    for &index in indices {
        let mut picked = None;
        let mut next_index = 0;

        expand(problem, curr.clone(), &mut |next| {
            if next_index == index {
                assert!(picked.is_none());
                picked = Some(next);
            }
            next_index += 1;
        });

        prev = Some(curr);
        curr = picked.ok_or(())?;
    }

    let mut children = vec![];
    expand(problem, curr.clone(), &mut |c| {
        // TODO remove check once expand stops generating duplicates
        if !children.contains(&c) {
            children.push(c)
        }
    });

    Ok((prev, curr, children))
}

fn build_html(problem: &Problem, indices: &[u64], parent_state: Option<&State>, state: &State, children: &[State]) -> String {
    assert_eq!(indices.is_empty(), parent_state.is_none());

    let mut svg = Vec::new();
    state.write_svg_to(problem, &mut svg).unwrap();
    let svg = String::from_utf8(svg).unwrap();

    let str_summary = state.summary_string(problem);
    let html_summary = format!(r#"<div id="text" style="white-space: pre; font-family: monospace">{str_summary}</div>"#);

    let state_actions_len = state.actions_taken.len();

    let mut possible_actions = HashSet::new();
    for child in children {
        let child_non_time_actions = &child.actions_taken[state_actions_len..child.actions_taken.len() - 1];
        possible_actions.extend(
            child_non_time_actions.iter()
                .map(|a| {
                    assert!(!matches!(a.inner, Action::Wait(_)));
                    a.inner
                })
        );
    }
    let possible_actions = possible_actions.iter().sorted().collect_vec();

    let action_cols = possible_actions.iter().map(|a| {
        let s = match a {
            Action::Wait(_) => unreachable!(),
            Action::Core(alloc) => format!("A{}", alloc.to_index()),
            Action::Channel(ActionChannel { channel, value }) => format!("C{}V{}", channel.to_index(), value.to_index()),
            Action::Drop(ActionDrop { value, mem }) => format!("D{}M{}", value.to_index(), mem.to_index()),
        };
        format!("<th>{s}</th>")
    }).join("");

    let filter_cols = (0..possible_actions.len())
        .map(|i| format!("<td><input id=\"filter_{i}\" type=\"checkbox\" onclick=\"update();\"/></td>"))
        .join("");
    let enable_cols = (0..possible_actions.len())
        .map(|i| format!("<td><input id=\"enable_{i}\"type=\"checkbox\" onclick=\"update();\"/></td>"))
        .join("");

    let mut html_children = String::new();
    let f = &mut html_children;

    writeln!(f, "<table class=\"table\">").unwrap();
    writeln!(f, "<tr><th>Link</th>{action_cols}<th>Time</th></tr>").unwrap();
    if let Some(parent_state) = parent_state {
        let delta = (parent_state.curr_time - state.curr_time).0;
        let dummy_cols = "<td></td>".repeat(possible_actions.len());
        writeln!(f, "<tr><td><a href=\"../\">back</a></td>{dummy_cols}<td>{delta}</td></tr>").unwrap();
    }
    writeln!(f, "<tr><td>Filter value</td>{filter_cols}<td></td></tr>").unwrap();
    writeln!(f, "<tr><td>Filter enable</td>{enable_cols}<td></td></tr>").unwrap();

    // TODO separate rendering for mandatory actions (eg. dead value drops)
    //   maybe extend to all actions that don't have alternatives?
    for (child_index, child) in enumerate(children) {
        let child_actions = &child.actions_taken[state_actions_len..];

        let mut hits = 0;
        let checked_cols = possible_actions.iter().enumerate().map(|(action_index, &a)| {
            let hit = match child_actions.iter().filter(|n| &n.inner == a).count() {
                0 => false,
                1 => true,
                _ => unreachable!(),
            };
            let s = if hit {
                hits += 1;
                format!("<input id=\"check_{}_{}\" type=\"checkbox\" checked disabled/>", child_index, action_index)
            } else {
                "".to_string()
            };
            format!("<td>{s}</td>")
        }).join("");
        assert_eq!(hits, child_actions.len() - 1);

        let delta = match child.actions_taken.last().unwrap().inner {
            Action::Wait(delta) => delta.0,
            _ => unreachable!(),
        };

        writeln!(f, "<tr id=\"row_{child_index}\"><td><a href=\"./{child_index}/\">{child_index}</a></td>{checked_cols}<td>{delta}</td></tr>").unwrap();
    }
    writeln!(f, "</table>").unwrap();

    let child_count = children.len();
    let action_count = possible_actions.len();

    format!(
        r#"
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <style type="text/css">
                    .table {{ background-color:#eee;border-collapse:collapse; }}
                    .table th {{ background-color:#222;color:white; }}
                    .table td, .table th {{ padding:5px;border:1px solid #000; }}
                    .table td {{ text-align: center; }}
                    .table tr td:first-child {{ text-align: right; }}
                    .table tr td:last-child {{ text-align: right; }}
                </style>
            </head>
            <body>
                {svg}
                {html_summary}
                {html_children}
            </body>
            <script>
                function update() {{
                    for (let row = 0; row < {child_count}; row++) {{
                        let match = true;
                        for (let col = 0; col < {action_count}; col++) {{
                            const enabled = document.getElementById(`enable_${{col}}`).checked;
                            const expected = document.getElementById(`filter_${{col}}`).checked;
                            const actual = document.getElementById(`check_${{row}}_${{col}}`) !== null;
                            if (enabled && (expected !== actual)) {{
                                match = false;
                                break;
                            }}
                        }}
                
                        const elem_row = document.getElementById(`row_${{row}}`);
                        if (match) {{
                            elem_row.style.display = "table-row";
                        }} else {{
                            elem_row.style.display = "none";
                        }}
                    }}
                }}
            </script>
        </html>
        "#,
    )
}