use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::core::problem::Problem;
use crate::core::schedule::{Action, TimeRange};
use crate::core::state::State;

impl State {
    pub fn write_svg_to<F: Write>(&self, problem: &Problem, mut f: F) -> std::io::Result<()> {
        let row_count = problem.hardware.groups().len();
        let time_max = self.minimum_time;

        let row_height = 70.0;
        let padding_left = 150.0;
        let padding_right = 50.0;
        let padding_ver = 50.0;
        let tick_size = 10.0;
        let time_div = 40.0;

        let figure_height = row_height * row_count as f64 + 2.0 * padding_ver;
        let figure_width = time_max / time_div + padding_left + padding_right;

        let time_to_x = |time: f64| padding_left + time / time_div;
        let row_to_y = |row: usize| padding_ver + row as f64 * row_height;

        writeln!(
            f,
            "<svg xmlns='http://www.w3.org/2000/svg' width='{}pt' height='{}pt' viewBox='0 0 {} {}'>",
            figure_width, figure_height, figure_width, figure_height
        )?;
        // writeln!(f, "<style>text {{ font-size: 20px; }}</style>")?;

        // grey background
        writeln!(f, "<rect width='100%' height='100%' fill='lightgrey' />")?;

        // bounding box
        writeln!(
            f,
            "<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='none' stroke='black' />",
            x = padding_left,
            y = padding_ver,
            w = figure_width - padding_left - padding_right,
            h = figure_height - 2.0 * padding_ver,
        )?;

        // vertical labels and lines
        for group in problem.hardware.groups() {
            let group_info = &problem.hardware.group_info[group.0];
            let y = row_to_y(group.0);
            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='left' font-weight='bold'>{t}</text>",
                x = padding_left / 10.0,
                y = y + row_height / 2.0,
                t = group_info.id,
            )?;

            if group.0 != 0 {
                writeln!(
                    f,
                    "<line x1='{x1}' y1='{y}' x2='{x2}' y2='{y}' stroke='grey' />",
                    x1 = padding_left,
                    x2 = figure_width - padding_right,
                    y = y,
                )?;
            }
        }

        // vertical lines and labels
        for t in pick_axis_ticks(time_max, 20) {
            let x = time_to_x(t as f64);
            writeln!(
                f,
                "<line x1='{x}' y1='{y1}' x2='{x}' y2='{y2}' stroke='grey' stroke-dasharray='5,5' />",
                x = x,
                y1 = padding_ver,
                y2 = figure_height - padding_ver + tick_size,
            )?;
            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='hanging' text-anchor='middle'>{t}</text>",
                x = x,
                y = figure_height - padding_ver + tick_size * 2.0,
                t = t,
            )?;
        }

        // draw action rectangles
        let rect = |f: &mut F, row, t_min, t_max, color, text: &str| {
            writeln!(
                f,
                "<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='none' stroke='{c}' />",
                x = time_to_x(t_min),
                y = row_to_y(row),
                w = time_to_x(t_max) - time_to_x(t_min),
                h = row_height,
                c = color,
            )?;

            // centered text
            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='middle' transform='rotate(20 {x} {y})'>{t}</text>",
                x = time_to_x((t_min + t_max) / 2.0),
                y = row_to_y(row) + row_height / 2.0,
                t = text,
            )?;

            Ok::<(), std::io::Error>(())
        };

        for action in &self.actions_taken {
            // comment action debug string
            writeln!(f, "<!-- {:?} -->", action)?;

            match action {
                Action::Wait(_) => {
                    // don't draw anything
                }
                Action::Core(action) => {
                    let alloc_info = &problem.allocation_info[action.alloc.0];
                    let node_info = &problem.graph.node_info[alloc_info.node.0];

                    let row = alloc_info.group.0;
                    let text = format!("{}\n{}", node_info.id, alloc_info.id);
                    rect(&mut f, row, action.time.start, action.time.end, "green", &text)?;
                }
                Action::Channel(action) => {
                    let channel_info = &problem.hardware.channel_info[action.channel.0];
                    let node_info = &problem.graph.node_info[action.value.0];

                    let row = channel_info.group.0;
                    rect(&mut f, row, action.time.start, action.time.end, "darkorange", &node_info.id)?;
                }
            }
        }

        // vertical red dashed line at current time
        writeln!(
            f,
            "<line x1='{x}' y1='0' x2='{x}' y2='{h}' stroke='red' stroke-dasharray='5,5' />",
            x = time_to_x(self.curr_time),
            h = figure_height
        )?;

        writeln!(f, "</svg>")?;

        Ok(())
    }

    pub fn write_svg_to_file(&self, problem: &Problem, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        self.write_svg_to(problem, &mut f)?;
        f.flush()?;
        drop(f);
        std::process::exit(0);
        Ok(())
    }
}

fn pick_axis_ticks(max_value: f64, _max_ticks: usize) -> Vec<f64> {
    // TODO improve this
    let mut ticks = Vec::new();
    let mut curr = 0.0;
    while curr <= max_value {
        ticks.push(curr);
        curr += 5_000.0;
    }
    if curr != max_value {
        ticks.push(max_value);
    }
    ticks
}