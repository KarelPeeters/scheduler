use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::core::problem::Problem;
use crate::core::schedule::Action;
use crate::core::state::State;

impl State {
    pub fn write_svg_to<F: Write>(&self, problem: &Problem, mut f: F) -> std::io::Result<()> {
        let row_count = problem.hardware.cores().len() + problem.hardware.channels().len();
        let time_max = self.minimum_time;

        let row_height = 100.0;
        let horizontal_padding_frac = 0.05;
        let figure_height = row_height * (2 + row_count) as f64;
        let time_div = time_max / (1.0 - 2.0 * horizontal_padding_frac);

        writeln!(f, "<svg xmlns='http://www.w3.org/2000/svg' width='1200pt' height='{}pt' viewBox='0 0 1200 {}'>", figure_height, figure_height)?;
        // writeln!(f, "<style>text {{ font-size: 20px; }}</style>")?;

        // grey background
        writeln!(f, "<rect width='100%' height='100%' fill='lightgrey' />")?;

        // bounding box
        writeln!(
            f,
            "<rect x='{}%' y='{}' width='{}%' height='{}' fill='none' stroke='black' />",
            100.0 * horizontal_padding_frac,
            row_height,
            100.0 * (1.0 - 2.0 * horizontal_padding_frac),
            figure_height - 2.0 * row_height
        )?;
        // split between cores and channels
        writeln!(
            f,
            "<line x1='{}%' y1='{}' x2='{}%' y2='{}' stroke='black' />",
            100.0 * horizontal_padding_frac,
            (problem.hardware.cores().len() as f64 + 1.0) * row_height,
            100.0 * (1.0 - horizontal_padding_frac),
            (problem.hardware.cores().len() as f64 + 1.0) * row_height
        )?;

        // draw action rectangles
        let rect = |f: &mut F, row, t_min, t_max, color, text| {
            writeln!(
                f,
                "<rect x='{}%' y='{}' width='{}%' height='{}' fill='none' stroke='{}' />",
                100.0 * (t_min / time_div + horizontal_padding_frac),
                (row + 1) as f64 * row_height,
                100.0 * ((t_max - t_min) / time_div),
                row_height,
                color,
            )?;

            // centered text
            writeln!(
                f,
                "<text x='{}%' y='{}' dominant-baseline='middle' text-anchor='middle'>{}</text>",
                100.0 * ((t_min + t_max) / 2.0 / time_div + horizontal_padding_frac),
                (row + 1) as f64 * row_height + row_height / 2.0,
                text,
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
                    // plot core
                    let alloc_info = &problem.allocation_info[action.alloc.0];
                    let node_info = &problem.graph.node_info[alloc_info.node.0];

                    let row = alloc_info.core.0;
                    rect(&mut f, row, action.time_start, action.time_end, "green", &node_info.id)?;
                }
                Action::Channel(action) => {
                    let row = problem.hardware.cores().len() + action.channel.0;
                    let node_info = &problem.graph.node_info[action.value.0];

                    rect(&mut f, row, action.time_start, action.time_end, "darkorange", &node_info.id)?;
                }
            }
        }

        // vertical red dashed line at current time
        writeln!(
            f,
            "<line x1='{}%' y1='0' x2='{}%' y2='{}' stroke='red' stroke-dasharray='5,5' />",
            100.0 * (self.curr_time / time_div + horizontal_padding_frac),
            100.0 * (self.curr_time / time_div + horizontal_padding_frac),
            figure_height
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
}