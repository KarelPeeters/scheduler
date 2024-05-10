use std::collections::{HashSet, VecDeque};
use std::fmt::Write;

use itertools::{enumerate, Itertools};
use rouille::{Request, Response};

use crate::core::expand::expand;
use crate::core::problem::Problem;
use crate::core::schedule::{Action, ActionChannel, ActionDrop};
use crate::core::solve_recurse::RecurseCache;
use crate::core::state::State;
use crate::core::wrapper::TypedIndex;

pub fn main_server(problem: Problem, recurse_cache: Option<RecurseCache>) {
    println!("Starting server");
    rouille::start_server("localhost:8000", move |request| {
        handle_request(&problem, recurse_cache.as_ref(), request)
    });
}

fn handle_request(problem: &Problem, cache: Option<&RecurseCache>, request: &Request) -> Response {
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
            let (parent, state) = match pick_state(problem, &indices) {
                Ok(state) => state,
                Err(_) => return Response::empty_404(),
            };

            let children = gen_children(problem, &state);
            Response::html(build_html(problem, cache, &indices, parent.as_ref(), &state, &children))
        }
        _ => Response::empty_404(),
    };

    println!("{:?}", r);

    r
}

fn pick_state(problem: &Problem, indices: &[u64]) -> Result<(Option<State>, State), ()> {
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

    Ok((prev, curr))
}

fn gen_children(problem: &Problem, state: &State) -> Vec<State> {
    let mut children = vec![];
    expand(problem, state.clone(), &mut |c| {
        // TODO remove check once expand stops generating duplicates
        if !children.contains(&c) {
            children.push(c)
        }
    });
    children
}

fn indices_for_target(problem: &Problem, target: &State) -> Vec<u64> {
    let mut result = vec![];
    let mut curr: State = State::new(problem);

    loop {
        if curr.actions_taken.len() == target.actions_taken.len() {
            break;
        }

        let children = gen_children(problem, &curr);
        let mut found = false;
        for (child_index, child) in enumerate(children.into_iter()) {
            if target.actions_taken.get(..child.actions_taken.len()) == Some(&child.actions_taken[..]) {
                result.push(child_index as u64);
                curr = child;
                found = true;
                break;
            }
        }

        assert!(found);
    }

    assert_eq!(&pick_state(problem, &result).unwrap().1.actions_taken, &target.actions_taken);

    result
}

fn build_html(problem: &Problem, cache: Option<&RecurseCache>, indices: &[u64], parent_state: Option<&State>, state: &State, children: &[State]) -> String {
    assert_eq!(indices.is_empty(), parent_state.is_none());

    let mut svg = Vec::new();
    state.write_svg_to(problem, &mut svg).unwrap();
    let svg = String::from_utf8(svg).unwrap();

    let mut str_frontier = String::new();
    if let Some(cache) = cache {
        let f = &mut str_frontier;
        writeln!(f, "Cache:").unwrap();
        writeln!(f, "  Frontier:").unwrap();
        if let Some(entry) = cache.get(&state.achievement(problem)) {
            for (&c, _) in entry.frontier.iter_arbitrary() {
                writeln!(f, "    {:?} -> {:?}", c, c + state.current_cost()).unwrap();
            }
            // writeln!(f, "  Example actions:").unwrap();
            // for &a in &entry.example_state.actions_taken {
            //     writeln!(f, "    {:?}", a).unwrap();
            // }
            // let indices = indices_for_target(problem, &entry.example_state);
            // writeln!(f, "    {}", indices.iter().map(|x| x.to_string()).join("/")).unwrap();
        } else {
            writeln!(f, "  none").unwrap();
        }
        writeln!(f).unwrap();
    };

    let str_summary = state.summary_string(problem);
    let html_summary = format!(r#"<div id="text" style="white-space: pre; font-family: monospace">{str_frontier}{str_summary}</div>"#);

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
    writeln!(f, "<tr><th>Link</th>{action_cols}<th>Time</th><th>Energy</th></tr>").unwrap();
    if let Some(parent_state) = parent_state {
        let delta_time = (parent_state.curr_time - state.curr_time).0;
        let delta_energy = (parent_state.curr_energy - state.curr_energy).0;
        let dummy_cols = "<td></td>".repeat(possible_actions.len());
        writeln!(f, "<tr><td><a href=\"../\">back</a></td>{dummy_cols}<td>{delta_time}</td><td>{delta_energy}</td></tr>").unwrap();
    }
    writeln!(f, "<tr><td>Filter value</td>{filter_cols}<td></td><td></td></tr>").unwrap();
    writeln!(f, "<tr><td>Filter enable</td>{enable_cols}<td></td><td></td></tr>").unwrap();

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

        let delta_time = (child.curr_time - state.curr_time).0;
        let delta_energy = (child.curr_energy - state.curr_energy).0;
        writeln!(f, "<tr id=\"row_{child_index}\"><td><a href=\"./{child_index}/\">{child_index}</a></td>{checked_cols}<td>{delta_time}</td><td>{delta_energy}</td></tr>").unwrap();
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