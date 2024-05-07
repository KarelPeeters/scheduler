use itertools::{enumerate, Itertools};
use rouille::{Request, Response};
use std::fmt::Write;

use crate::core::expand::expand;
use crate::core::problem::Problem;
use crate::core::state::State;

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
    expand(problem, curr.clone(), &mut |c| children.push(c));

    Ok((prev, curr, children))
}

fn build_html(problem: &Problem, indices: &[u64], parent_state: Option<&State>, state: &State, children: &[State]) -> String {
    assert_eq!(indices.is_empty(), parent_state.is_none());

    let mut svg = Vec::new();
    state.write_svg_to(problem, &mut svg).unwrap();
    let svg = String::from_utf8(svg).unwrap();

    let mut html_children = String::new();
    let f = &mut html_children;

    writeln!(f, "<table>").unwrap();
    writeln!(f, "<tr><th>Actions</th><th>Time</th><th>Link</th></tr>").unwrap();
    if let Some(parent_state) = parent_state {
        let delta = (parent_state.curr_time - state.curr_time).0;
        writeln!(f, "<tr><td>back</td><td>{delta}</td><td><a href=\"../\">back</a></td></tr>").unwrap();
    }
    for (child_index, child) in enumerate(children) {
        let actions_str = child.actions_taken.iter().map(|a| format!("{:?}", a.inner)).join("<p>");
        let delta = (child.curr_time - state.curr_time).0;
        writeln!(f, "<tr><td>{actions_str}</td><td>{delta}</td><td><a href=\"./{child_index}/\">{child_index}</a></td></tr>").unwrap();
    }
    writeln!(f, "</table>").unwrap();

    format!(
        r#"
        <!DOCTYPE html>
        <html lang="en">
        {svg}
        {html_children}
        </html>
        "#,
    )
}