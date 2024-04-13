use std::convert::identity;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use itertools::Itertools;

use crate::core::frontier::Frontier;
use crate::core::problem::Problem;
use crate::core::schedule::Action;
use crate::core::state::{Cost, State};

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

        // background
        writeln!(f, "<rect width='100%' height='100%' fill='white' />")?;

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
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='start' font-weight='bold'>{t}</text>",
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
        for (t, label) in pick_axis_ticks(time_max, 20) {
            let x = time_to_x(t);
            if t != 0.0 {
                writeln!(
                    f,
                    "<line x1='{x}' y1='{y1}' x2='{x}' y2='{y2}' stroke='grey' stroke-dasharray='5,5' />",
                    x = x,
                    y1 = padding_ver,
                    y2 = figure_height - padding_ver + tick_size,
                )?;
            }
            if label {
                writeln!(
                    f,
                    "<text x='{x}' y='{y}' dominant-baseline='hanging' text-anchor='middle'>{t}</text>",
                    x = x,
                    y = figure_height - padding_ver + tick_size * 2.0,
                    t = t,
                )?;
            }
        }

        // draw action rectangles
        let rect = |f: &mut F, row, t_min, t_max, color, text: &str| {
            writeln!(
                f,
                "<rect x='{x}' y='{y}' width='{w}' height='{h}' stroke='{c}' stoke-width='1pt' fill='{c}' fill-opacity='0.2' />",
                x = time_to_x(t_min),
                y = row_to_y(row) + 0.5,
                w = time_to_x(t_max) - time_to_x(t_min),
                h = row_height - 1.0,
                c = color,
            )?;

            // centered text
            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='middle' transform='rotate(30 {x} {y})'>{t}</text>",
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
        Ok(())
    }

    pub fn summary_string(&self, problem: &Problem) -> String {
        fn inner(s: &State, problem: &Problem, f: &mut String) -> std::fmt::Result {
            writeln!(f, "Basics:")?;
            writeln!(f, "  curr_time={}", s.curr_time)?;
            writeln!(f, "  curr_energy={}", s.curr_energy)?;
            writeln!(f, "  minimum_time={}", s.minimum_time)?;
            writeln!(f)?;

            writeln!(f, "Memory:")?;
            for mem in problem.hardware.memories() {
                let info = &problem.hardware.mem_info[mem.0];
                writeln!(f, "  memory '{}': used {}/{:?}", &info.id, s.mem_space_used(problem, mem), info.size_bits)?;

                // iterate ourselves to keep order
                for value in problem.graph.nodes() {
                    if let Some(state) = s.state_memory_node[mem.0].get(&value) {
                        let value_info = &problem.graph.node_info[value.0];
                        writeln!(f, "    contains '{}' with size {}, state {:?}", value_info.id, value_info.size_bits, state)?;
                    }
                }
            }
            writeln!(f)?;

            writeln!(f, "Triggers:")?;
            writeln!(f, "  everything: {}", s.trigger_everything)?;

            let trigger_group_free_str = s.trigger_group_free.iter().copied().positions(identity)
                .map(|g| format!("group '{}'", problem.hardware.group_info[g].id))
                .join(", ");
            writeln!(f, "  group_free: {}", trigger_group_free_str)?;

            let trigger_value_mem_available_str = s.trigger_value_mem_available.iter().enumerate().flat_map(|(v, mems)| {
                mems.iter().copied().positions(identity).map(move |m| (v, m))
            }).map(|(v, m)| {
                let value_info = &problem.graph.node_info[v];
                let mem_info = &problem.hardware.mem_info[m];
                format!("value '{}' in memory '{}'", value_info.id, mem_info.id)
            }).join(", ");
            writeln!(f, "  value_mem_available: {}", trigger_value_mem_available_str)?;

            let trigger_value_mem_unlocked = s.trigger_value_mem_unlocked.iter().enumerate().flat_map(|(v, mems)| {
                mems.iter().copied().positions(identity).map(move |m| (v, m))
            }).map(|(v, m)| {
                let value_info = &problem.graph.node_info[v];
                let mem_info = &problem.hardware.mem_info[m];
                format!("value '{}' in memory '{}'", value_info.id, mem_info.id)
            }).join(", ");
            writeln!(f, "  value_mem_unlocked: {}", trigger_value_mem_unlocked)?;

            let trigger_mem_usage_decreased_str = s.trigger_mem_usage_decreased.iter().enumerate().filter_map(|(v, mems)| {
                mems.map(move |delta| (v, delta))
            }).map(|(v, delta)| {
                let mem_info = &problem.hardware.mem_info[v];
                format!("memory '{}' from {} to {}", mem_info.id, delta.0, delta.1)
            }).join(", ");
            writeln!(f, "  mem_usage_decreased: {}", trigger_mem_usage_decreased_str)?;

            Ok(())
        }

        let mut result = String::new();
        inner(self, problem, &mut result).unwrap();
        result
    }
}

fn pick_axis_ticks(max_value: f64, _max_ticks: usize) -> Vec<(f64, bool)> {
    // TODO improve this
    let step_size = 5_000.0;

    let mut ticks = Vec::new();
    let mut curr = 0.0;
    while curr <= max_value {
        ticks.push((curr, true));
        curr += step_size;
    }

    if curr != max_value {
        if let Some(last) = ticks.last_mut() {
            if max_value < last.0 + step_size / 3.0 {
                last.1 = false;
            }
        }
        ticks.push((max_value, true));
    }

    ticks
}

impl<V> Frontier<Cost, V> {
    pub fn write_svg_to(&self, old: &[Cost], mut f: impl Write) -> std::io::Result<()> {
        let curr = self.iter_arbitrary().map(|(&c, _)| c).collect::<Vec<_>>();
        
        let scatter_old = poloto::build::plot("old")
            .scatter(old.iter().filter(|c| !curr.contains(c)).map(|c| (c.time, c.energy)));
        let scatter_curr = poloto::build::plot("curr")
            .scatter(curr.iter().map(|c| (c.time, c.energy)));
        
        let plots = poloto::plots!(scatter_old, scatter_curr);

        let result = poloto::frame_build().data(plots)
            .build_and_label(("Pareto front", "Time", "Energy"))
            .append_to(poloto::header().light_theme())
            .render_string().unwrap();
        write!(f, "{}", result)
    }

    pub fn write_svg_to_file(&self, old: &[Cost], path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        self.write_svg_to(old, &mut f)?;
        f.flush()?;
        drop(f);
        Ok(())
    }
}