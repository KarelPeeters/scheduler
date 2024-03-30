use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub struct GraphViz {
    source: String,
}

impl GraphViz {
    pub fn new() -> Self {
        GraphViz { source: String::new() }
    }

    pub fn push(&mut self, line: impl AsRef<str>) {
        self.source.push_str("    ");
        self.source.push_str(line.as_ref());
        self.source.push('\n');
    }

    pub fn export(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path.parent().unwrap())?;

        let path_dot = path.with_extension("dot");
        let mut file = File::create(&path_dot)?;
        writeln!(file, "digraph G {{")?;
        file.write_all(self.source.as_bytes())?;
        writeln!(file, "}}")?;

        let output = Command::new("dot")
            .arg(format!("-T{}", path.extension().unwrap().to_str().unwrap()))
            .arg(path_dot)
            .arg("-o")
            .arg(path)
            .stderr(std::process::Stdio::piped())
            .output()?;
        if !output.status.success() {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, String::from_utf8_lossy(&output.stderr).trim()));
        }

        Ok(())
    }

    pub fn table(title: &str, rows: Vec<(&str, String)>) -> String {
        let mut table = format!("<TABLE BORDER=\"0\" COLUMNS=\"*\" ROWS=\"*\"><TR><TD colspan=\"2\"><B>{}</B></TD></TR>", title);
        for (a, b) in rows {
            table += &format!("<TR><TD>{}</TD><TD>{}</TD></TR>", a, b);
        }
        table += "</TABLE>";
        table
    }

    pub fn html(content: &String) -> String {
        format!("<{content}>")
    }
}
