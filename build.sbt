name := "Simple Machine Learning"

// set version to environment variable $tag 
version := sys.props.getOrElse("tag", default = "0.0.0")

organization := "com.github.rehei"

scalaVersion := "2.11.7"

resolvers += "snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
resolvers += "staging" at "https://oss.sonatype.org/content/repositories/staging"
resolvers += "releases" at "https://oss.sonatype.org/content/repositories/releases"
resolvers += Resolver.bintrayRepo("rehei", "maven")

unmanagedResourceDirectories in Test <+= (baseDirectory) { _ / "src/main/webapp" }

scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature")

libraryDependencies ++= {
  Seq(
    "nz.ac.waikato.cms.weka" % "weka-stable" % "3.6.13"
  )
}

EclipseKeys.withSource := true

