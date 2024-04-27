use std::convert::identity;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use itertools::{Itertools, zip_eq};

use crate::core::frontier::Frontier;
use crate::core::problem::{Memory, Problem};
use crate::core::schedule::Action;
use crate::core::state::{Cost, State};
use crate::core::wrapper::Time;

impl State {
    pub fn write_svg_to<F: Write>(&self, problem: &Problem, mut f: F) -> std::io::Result<()> {
        let hardware = &problem.hardware;
        let graph = &problem.graph;

        let row_count = hardware.groups().len() + hardware.memories().len();
        let time_max = self.minimum_time.0 as f64;

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
        let mem_to_y_delta = |value: u64, limit: u64| row_height * (1.0 - value as f64 / limit as f64);

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
        for group in hardware.groups() {
            let group_info = &hardware.group_info[group.0];
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
                Action::Wait(_) | Action::Drop(_) => {
                    // don't draw anything
                }
                Action::Core(action) => {
                    let alloc_info = &problem.allocation_info[action.alloc.0];
                    let node_info = &graph.node_info[alloc_info.node.0];

                    let row = alloc_info.group.0;
                    let text = format!("#{} {} {}", action.alloc.0, node_info.id, alloc_info.id);
                    rect(&mut f, row, action.time.start.0 as f64, action.time.end.0 as f64, "green", &text)?;
                }
                Action::Channel(action) => {
                    let channel_info = &hardware.channel_info[action.channel.0];
                    let node_info = &graph.node_info[action.value.0];

                    let row = channel_info.group.0;
                    rect(&mut f, row, action.time.start.0 as f64, action.time.end.0 as f64, "darkorange", &node_info.id)?;
                }
            }
        }

        // collect memory plot info
        let mut mem_time_bits_used = vec![vec![(0.0, 0)]; hardware.memories().len()];
        for (node, mem) in zip_eq(&graph.inputs, &problem.input_placements) {
            mem_time_bits_used[mem.0][0].1 += graph.node_info[node.0].size_bits;
        }
        let mut add_size_delta = |mem: Memory, time: Time, delta: i64| {
            let vec = &mut mem_time_bits_used[mem.0];
            let &(_, prev_used) = vec.last().unwrap();
            vec.push((time.0 as f64, prev_used));
            vec.push((time.0 as f64, prev_used.checked_add_signed(delta).unwrap()));
        };
        for action in &self.actions_taken {
            match action {
                Action::Wait(_) => {
                    // no memory implications
                }
                Action::Core(action) => {
                    let alloc_info = &problem.allocation_info[action.alloc.0];
                    let node_info = &graph.node_info[alloc_info.node.0];
                    add_size_delta(alloc_info.output_memory, action.time.start, node_info.size_bits as i64);
                }
                Action::Channel(action) => {
                    let channel_info = &hardware.channel_info[action.channel.0];
                    let node_info = &graph.node_info[action.value.0];
                    add_size_delta(channel_info.mem_dest, action.time.start, node_info.size_bits as i64);
                }
                Action::Drop(action) => {
                    let node_info = &graph.node_info[action.value.0];
                    add_size_delta(action.mem, action.time, -(node_info.size_bits as i64));
                }
            }
        }
        for mem in hardware.memories() {
            add_size_delta(mem, self.minimum_time, 0);
        }

        // draw memory plots
        for mem in hardware.memories() {
            let mem_info = &hardware.mem_info[mem.0];
            let y = row_to_y(mem.0 + hardware.groups().len());

            let history = &mem_time_bits_used[mem.0];
            let max_bits = history.iter().map(|&(_, b)| b).max().unwrap_or(1);
            let mut plot_limit = mem_info.size_bits.unwrap_or(max_bits);
            if plot_limit == 0 {
                plot_limit = 1;
            }

            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='start' font-weight='bold'>{t}</text>",
                x = padding_left / 10.0,
                y = y + row_height / 2.0,
                t = mem_info.id,
            )?;

            let limit_str = mem_info.size_bits.map_or("".to_string(), |s| format!("/{s}"));
            writeln!(
                f,
                "<text x='{x}' y='{y}' dominant-baseline='middle' text-anchor='start'>{t}</text>",
                x = padding_left / 10.0,
                y = y + row_height / 2.0 + 20.0,
                t = format!("peak {max_bits}{limit_str}"),
            )?;

            writeln!(
                f,
                "<line x1='{x1}' y1='{y}' x2='{x2}' y2='{y}' stroke='grey' />",
                x1 = padding_left,
                x2 = figure_width - padding_right,
                y = y,
            )?;

            // plot lines
            for (&(prev_t, prev_b), &(next_t, next_b)) in history.iter().tuple_windows() {
                let x1 = time_to_x(prev_t);
                let x2 = time_to_x(next_t);
                let d1 = mem_to_y_delta(prev_b, plot_limit);
                let y1 = y + d1;
                let y2 = y + mem_to_y_delta(next_b, plot_limit);

                writeln!(f, "<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='blue'/>")?;

                if y1 == y2 {
                    // rectangle that drops to the bottom
                    let w = x2 - x1;
                    let h = row_height - d1;
                    writeln!(f, "<rect x='{x1}' y='{y1}' width='{w}' height='{h}' fill='blue' fill-opacity='0.2'/>")?;
                }
            }
        }

        // vertical red dashed line at current time
        writeln!(
            f,
            "<line x1='{x}' y1='0' x2='{x}' y2='{h}' stroke='red' stroke-dasharray='5,5' />",
            x = time_to_x(self.curr_time.0 as f64),
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
            let hardware = &problem.hardware;
            let graph = &problem.graph;

            writeln!(f, "Basics:")?;
            writeln!(f, "  curr_time={}", s.curr_time.0)?;
            writeln!(f, "  curr_energy={}", s.curr_energy.0)?;
            writeln!(f, "  minimum_time={}", s.minimum_time.0)?;
            writeln!(f)?;

            writeln!(f, "Memory:")?;
            for mem in hardware.memories() {
                let info = &hardware.mem_info[mem.0];
                writeln!(f, "  memory '{}': used {}/{:?}", &info.id, s.mem_space_used(problem, mem), info.size_bits)?;

                // iterate ourselves to keep order
                for value in graph.nodes() {
                    if let Some(state) = s.state_memory_node[mem.0].get(&value) {
                        let value_info = &graph.node_info[value.0];
                        writeln!(f, "    contains '{}' with size {}, state {:?}", value_info.id, value_info.size_bits, state)?;
                    }
                }
            }
            writeln!(f)?;

            writeln!(f, "Triggers:")?;
            writeln!(f, "  everything: {}", s.trigger_everything)?;

            let trigger_group_free_str = s.trigger_group_free.iter().copied().positions(identity)
                .map(|g| format!("group '{}'", hardware.group_info[g].id))
                .join(", ");
            writeln!(f, "  group_free: {}", trigger_group_free_str)?;

            let trigger_value_mem_available_str = s.trigger_value_mem_available.iter().enumerate().flat_map(|(v, mems)| {
                mems.iter().copied().positions(identity).map(move |m| (v, m))
            }).map(|(v, m)| {
                let value_info = &graph.node_info[v];
                let mem_info = &hardware.mem_info[m];
                format!("value '{}' in memory '{}'", value_info.id, mem_info.id)
            }).join(", ");
            writeln!(f, "  value_mem_available: {}", trigger_value_mem_available_str)?;

            let trigger_value_mem_unlocked = s.trigger_value_mem_unlocked_or_read.iter().enumerate().flat_map(|(v, mems)| {
                mems.iter().copied().positions(identity).map(move |m| (v, m))
            }).map(|(v, m)| {
                let value_info = &graph.node_info[v];
                let mem_info = &hardware.mem_info[m];
                format!("value '{}' in memory '{}'", value_info.id, mem_info.id)
            }).join(", ");
            writeln!(f, "  value_mem_unlocked: {}", trigger_value_mem_unlocked)?;

            let trigger_mem_usage_decreased_str = s.trigger_mem_usage_decreased.iter().enumerate().filter_map(|(v, mems)| {
                mems.map(move |delta| (v, delta))
            }).map(|(v, delta)| {
                let mem_info = &hardware.mem_info[v];
                format!("memory '{}' from {} to {}", mem_info.id, delta.0, delta.1)
            }).join(", ");
            writeln!(f, "  mem_usage_decreased: {}", trigger_mem_usage_decreased_str)?;

            let trigger_value_live_count_increased_str = s.trigger_value_live_count_increased.iter().enumerate().filter_map(|(v, &increased)| {
                if increased {
                    Some(format!("value '{}'", graph.node_info[v].id))
                } else {
                    None
                }
            }).join(", ");
            writeln!(f, "  value_live_count_increased: {}", trigger_value_live_count_increased_str)?;

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
            .scatter(old.iter().filter(|c| !curr.contains(c)).map(|c| (c.time.0 as f64, c.energy.0 as f64)));
        let scatter_curr = poloto::build::plot("curr")
            .scatter(curr.iter().map(|c| (c.time.0 as f64, c.energy.0 as f64)));
        
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