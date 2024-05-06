use itertools::{enumerate, Itertools};
use rouille::{Request, Response};
use std::fmt::Write;

use crate::core::expand::expand;
use crate::core::problem::Problem;
use crate::core::state::State;

pub fn main_server(problem: Problem) {
    rouille::start_server("localhost:80", move |request| {
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
            let (state, children) = match pick_state(problem, &indices) {
                Ok(state) => state,
                Err(_) => return Response::empty_404(),
            };

            Response::html(build_html(problem, &indices, &state, &children))
        }
        _ => Response::empty_404(),
    };

    println!("{:?}", r);

    r
}

fn pick_state(problem: &Problem, indices: &[u64]) -> Result<(State, Vec<State>), ()> {
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

        curr = picked.ok_or(())?;
    }

    let mut children = vec![];
    expand(problem, curr.clone(), &mut |c| children.push(c));

    Ok((curr, children))
}

fn build_html(problem: &Problem, indices: &[u64], state: &State, children: &[State]) -> String {
    let mut svg = Vec::new();
    state.write_svg_to(problem, &mut svg).unwrap();
    let svg = String::from_utf8(svg).unwrap();

    let mut html_children = String::new();
    let f = &mut html_children;

    if !indices.is_empty() {
        writeln!(f, "<a href=\"../\">back</a>").unwrap();
    }

    for (child_index, _) in enumerate(children) {
        writeln!(f, "<a href=\"./{child_index}/\">{child_index}</a>").unwrap();
    }

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